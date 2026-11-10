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

#include "itkCovariantVector.h"
#include "itkImportImageContainer.h"
#include "itkMacro.h"
#include "itkObjectFactoryBase.h"
#include "itkVersion.h"

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

#ifndef _WIN32
#  include <cerrno>
#  include <cstdlib>
#  include <cstring>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <unistd.h>
#endif

namespace itk
{

/** \class FileMappedScratch
 * \brief Blocks of memory mapped from files of a scratch directory.
 *
 * A block is the mapping of a file created in the directory and unlinked at once: the kernel
 * pages it in and out as memory allows, where an allocation stays in memory or in the swap,
 * and the file goes with its mapping, however the process ends.
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
    Blocks()[block] = bytes;
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
    munmap(block, found->second);
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
  static std::unordered_map<void *, std::size_t> &
  Blocks()
  {
    static std::unordered_map<void *, std::size_t> blocks;
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

} // end namespace itk

#endif
