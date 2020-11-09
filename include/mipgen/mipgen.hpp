#pragma once

#include <cstdint>

// Right now, this library is NOT THREAD SAFE.
// Specifically: A single context can only be used on one thread. Sharing a context across multiple threads requires external synchronization.
// In the future, I might improve this so a single context can be used concurrently.

namespace mipgen {

struct Context;

Context* create_context();
// Create context from existing instance, device and physical device. queue must be a queue capable of compute operations when MIPGEN_COMPUTE is true, otherwise it
// must be graphics capable
Context* create_context(uint64_t instance, uint64_t device, uint64_t phys_device, uint64_t queue, uint32_t family_index);
void free_context(Context* ctx);

enum class ImageFormat {
	RGBA8,
	sRGBA8
};

struct ImageInfo {
	// User pointer to the original image pixels
	void* pixels = nullptr;
	uint32_t extents[2]{ 0, 0 };
	ImageFormat format = ImageFormat::sRGBA8;
};

// Gets the amount of miplevels that will be generated from an image
uint32_t get_mip_count(ImageInfo const& image);
// Gets the total output buffer size required (in bytes).
uint32_t output_buffer_size(ImageInfo const& image);

// Generates miplevels for the image. Note that this call blocks the calling thread until the generation is complete.
// After this call, out_buffer will contain the pixels of the image with mip levels added.
// out_buffer must be at least output_buffer_size(image) large.
// This function returns the amount of bytes written to out_buffer.
// On error, out_buffer is left unmodified and the function returns zero.
uint32_t generate_mipmap(Context* ctx, ImageInfo const& image, void* out_buffer);

}