/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkFileMappedImageContainer_h
#define itkFileMappedImageContainer_h

#include "itkByteSwapper.h"
#include "itkCovariantVector.h"
#include "itkImageFileReader.h"
#include "itkImportImageContainer.h"
#include "itkMacro.h"
#include "itkObjectFactoryBase.h"
#include "itkVersion.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>

#ifndef _WIN32
#  include <cerrno>
#  include <cstdlib>
#  include <cstring>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace itk
{

/** \class FileMappedScratch
 * \brief Blocks of memory mapped from files of a scratch directory.
 *
 * A block is the mapping of a file created in the directory and unlinked at once: the kernel
 * pages it in and out as memory allows, where an allocation stays in memory or in the swap,
 * and the file goes with its mapping, however the process ends. The pixels of an image file
 * are mapped where they are.
 */
class FileMappedScratch
{
public:
  /** The directory the blocks are mapped from; empty until SetDirectory(). */
  static std::string &
  Directory()
  {
    static std::string directory;
    return directory;
  }

  /** A zero-filled block of `bytes` bytes. */
  static void *
  Allocate(const std::size_t bytes)
  {
#ifdef _WIN32
    // ponytail: POSIX mappings only; CreateFileMapping/MapViewOfFile when a Windows user needs it.
    itkGenericExceptionMacro("Images mapped from files are not available on Windows.");
#else
    std::string  path = Directory() + "/elastix_image_XXXXXX";
    const int    descriptor = mkstemp(path.data());
    const auto   fail = [&](const char * const what) {
      const std::string reason = std::strerror(errno);
      if (descriptor >= 0)
      {
        close(descriptor);
      }
      itkGenericExceptionMacro("Could not " << what << " " << bytes << " bytes in " << Directory() << ": " << reason);
    };
    if (descriptor < 0)
    {
      fail("create a file of");
    }
    unlink(path.c_str());
#  ifdef __APPLE__
    // No posix_fallocate there: the file is sized, and a full device faults at the first write of a page.
    const int error = ftruncate(descriptor, static_cast<off_t>(bytes)) == 0 ? 0 : errno;
#  else
    // Reserved on the device now: a full device is an error here, not a fault at the first write of a page.
    const int error = posix_fallocate(descriptor, 0, static_cast<off_t>(bytes));
#  endif
    if (error != 0)
    {
      errno = error;
      fail("reserve");
    }
    void * const block = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
    if (block == MAP_FAILED)
    {
      fail("map");
    }
    close(descriptor);
    const std::lock_guard<std::mutex> lock(Mutex());
    Blocks()[block] = { block, bytes };
    return block;
#endif
  }

  /** The `bytes` bytes of the file at `path` from `offset`, or null when the file does not hold them.
   * The mapping is private: its pages are read from the file as they are touched and dropped
   * under pressure without a write, and a write stays in memory.
   */
  static void *
  MapFile(const std::string & path, const std::size_t offset, const std::size_t bytes)
  {
#ifdef _WIN32
    return nullptr;
#else
    const int   descriptor = open(path.c_str(), O_RDONLY);
    struct stat status;
    if (descriptor < 0 || fstat(descriptor, &status) != 0 || static_cast<std::size_t>(status.st_size) < offset + bytes)
    {
      if (descriptor >= 0)
      {
        close(descriptor);
      }
      return nullptr;
    }
    const std::size_t lead = offset % static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    void * const      start = mmap(nullptr,
                              lead + bytes,
                              PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_NORESERVE,
                              descriptor,
                              static_cast<off_t>(offset - lead));
    close(descriptor);
    if (start == MAP_FAILED)
    {
      return nullptr;
    }
    void * const                      block = static_cast<char *>(start) + lead;
    const std::lock_guard<std::mutex> lock(Mutex());
    Blocks()[block] = { start, lead + bytes };
    return block;
#endif
  }

  /** Unmaps `block` when Allocate() returned it; tells whether it did. */
  static bool
  Release(void * const block)
  {
#ifdef _WIN32
    return false;
#else
    const std::lock_guard<std::mutex> lock(Mutex());
    const auto                        found = Blocks().find(block);
    if (found == Blocks().end())
    {
      return false;
    }
    munmap(found->second.first, found->second.second);
    Blocks().erase(found);
    return true;
#endif
  }

private:
  static std::mutex &
  Mutex()
  {
    static std::mutex mutex;
    return mutex;
  }
  /** The start and the length of the mapping of each block. */
  static std::unordered_map<void *, std::pair<void *, std::size_t>> &
  Blocks()
  {
    static std::unordered_map<void *, std::pair<void *, std::size_t>> blocks;
    return blocks;
  }
};


/** \class FileMappedImageContainer
 * \brief The pixel container of an image, its large buffers mapped from files.
 *
 * A buffer of FileMappedImageContainer::Threshold bytes or more is a block of
 * FileMappedScratch; a smaller one is allocated as ImportImageContainer does.
 */
template <typename TElement>
class ITK_TEMPLATE_EXPORT FileMappedImageContainer : public ImportImageContainer<SizeValueType, TElement>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(FileMappedImageContainer);

  using Self = FileMappedImageContainer;
  using Superclass = ImportImageContainer<SizeValueType, TElement>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;
  using typename Superclass::ElementIdentifier;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(FileMappedImageContainer);

  /** The size in bytes from which a buffer is mapped from a file. */
  static constexpr std::size_t Threshold = std::size_t{ 16 } << 20;

protected:
  FileMappedImageContainer() = default;
  ~FileMappedImageContainer() override { this->DeallocateManagedMemory(); }

  TElement *
  AllocateElements(const ElementIdentifier size, const bool useValueInitialization = false) const override
  {
    const std::size_t bytes = static_cast<std::size_t>(size) * sizeof(TElement);
    return bytes < Threshold ? Superclass::AllocateElements(size, useValueInitialization)
                             : static_cast<TElement *>(FileMappedScratch::Allocate(bytes));
  }

  void
  DeallocateManagedMemory() override
  {
    if (this->GetContainerManageMemory() && FileMappedScratch::Release(this->GetImportPointer()))
    {
      this->SetImportPointer(nullptr);
      this->SetCapacity(0);
      this->SetSize(0);
    }
    else
    {
      Superclass::DeallocateManagedMemory();
    }
  }
};


/** \class FileMappedImageContainerFactory
 * \brief Makes the images of the process keep their large buffers in files of a directory.
 *
 * MapImagesFrom() overrides the pixel containers of the scalar images elastix allocates (the
 * images, their pyramid levels, their B-spline coefficients, the masks) and of its gradient
 * images: a registration then runs on images larger than the memory, the kernel paging them,
 * and computes what it computes with its images in memory.
 */
class FileMappedImageContainerFactory : public ObjectFactoryBase
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(FileMappedImageContainerFactory);

  using Self = FileMappedImageContainerFactory;
  using Superclass = ObjectFactoryBase;
  using Pointer = SmartPointer<Self>;

  itkFactorylessNewMacro(Self);
  itkOverrideGetNameOfClassMacro(FileMappedImageContainerFactory);

  const char *
  GetITKSourceVersion() const override
  {
    return ITK_SOURCE_VERSION;
  }
  const char *
  GetDescription() const override
  {
    return "Image buffers mapped from files";
  }

  /** Maps the large image buffers allocated from now on from files of `directory`. */
  static void
  MapImagesFrom(const std::string & directory)
  {
    FileMappedScratch::Directory() = directory;
    ObjectFactoryBase::RegisterFactory(Self::New());
  }

protected:
  FileMappedImageContainerFactory()
  {
    this->Override<unsigned char>();
    this->Override<short>();
    this->Override<float>();
    this->Override<double>();
    this->Override<CovariantVector<double, 2>>();
    this->Override<CovariantVector<double, 3>>();
    this->Override<CovariantVector<double, 4>>();
  }

private:
  template <typename TElement>
  void
  Override()
  {
    this->RegisterOverride(typeid(ImportImageContainer<SizeValueType, TElement>).name(),
                           typeid(FileMappedImageContainer<TElement>).name(),
                           "Image buffer mapped from a file",
                           true,
                           CreateObjectFunction<FileMappedImageContainer<TElement>>::New());
  }
};



/** The image of the MetaImage file `fileName` with its pixels mapped from the file, or null when
 * they cannot be: no FileMappedImageContainerFactory, a compressed file, another pixel type or
 * byte order than the image's, pixels in several files, an image under the mapping threshold.
 * The caller then reads the file.
 */
template <typename TImage>
typename TImage::Pointer
MapImageFile(const std::string & fileName)
{
  using PixelType = typename TImage::PixelType;

  std::string extension = fileName.substr(std::min(fileName.size(), fileName.rfind('.') + 1));
  std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (FileMappedScratch::Directory().empty() || (extension != "mha" && extension != "mhd"))
  {
    return nullptr;
  }

  // The header is lines of "Key = Value" and ends with the line that tells where the pixels are.
  std::ifstream header(fileName, std::ios::binary);
  std::string   line;
  std::string   pixels;
  while (pixels.empty() && std::getline(header, line))
  {
    const auto equals = line.find(" = ");
    if (equals == std::string::npos)
    {
      return nullptr;
    }
    const std::string key = line.substr(0, equals);
    const std::string value = line.substr(equals + 3, line.find_last_not_of(" \r") - equals - 2);
    if ((key == "CompressedData" && value == "True") || (key == "HeaderSize" && value != "0"))
    {
      return nullptr;
    }
    if (key == "ElementDataFile")
    {
      pixels = value;
    }
  }
  if (pixels.empty() || pixels == "LIST" || pixels.find('%') != std::string::npos)
  {
    return nullptr;
  }
  const bool        local = pixels == "LOCAL" || pixels == "Local" || pixels == "local";
  const std::size_t offset = local ? static_cast<std::size_t>(header.tellg()) : 0;
  const std::string path = local ? fileName : fileName.substr(0, fileName.find_last_of("/\\") + 1) + pixels;

  const auto reader = ImageFileReader<TImage>::New();
  reader->SetFileName(fileName);
  reader->UpdateOutputInformation();
  const ImageIOBase &             io = *reader->GetImageIO();
  const typename TImage::Pointer  image = reader->GetOutput();
  const SizeValueType             numberOfPixels = image->GetLargestPossibleRegion().GetNumberOfPixels();
  const std::size_t               bytes = numberOfPixels * sizeof(PixelType);
  const IOByteOrderEnum           native =
    ByteSwapper<int>::SystemIsLittleEndian() ? IOByteOrderEnum::LittleEndian : IOByteOrderEnum::BigEndian;
  if (io.GetComponentType() != ImageIOBase::MapPixelType<PixelType>::CType || io.GetNumberOfComponents() != 1 ||
      io.GetNumberOfDimensions() != TImage::ImageDimension || io.GetByteOrder() != native ||
      offset % alignof(PixelType) != 0 || bytes < FileMappedImageContainer<PixelType>::Threshold ||
      dynamic_cast<FileMappedImageContainer<PixelType> *>(image->GetPixelContainer()) == nullptr)
  {
    return nullptr;
  }
  void * const block = FileMappedScratch::MapFile(path, offset, bytes);
  if (block == nullptr)
  {
    return nullptr;
  }
  image->DisconnectPipeline();
  image->SetBufferedRegion(image->GetLargestPossibleRegion());
  image->GetPixelContainer()->SetImportPointer(static_cast<PixelType *>(block), numberOfPixels, true);
  return image;
}

} // end namespace itk

#endif
