using BinaryBuilder, Pkg

name = "VulkanSift_jl"
version = v"0.1.0"

sources = [
    GitSource("https://github.com/prittjam/VulkanSift.git",
              "98faa3d8c31a026d17d52a0f02742945a225a8f5"),  # jl-wrapper branch
]

script = raw"""
cd $WORKSPACE/srcdir/VulkanSift*

cmake -B build -G Ninja \
    -DCMAKE_INSTALL_PREFIX=${prefix} \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DVULKANSIFT_LOAD_VK_AT_RUNTIME=ON \
    -DVULKANSIFT_BUILD_EXAMPLES=OFF \
    -DVULKANSIFT_BUILD_PERFS=OFF \
    -DVULKANSIFT_BUILD_JL_WRAPPER=ON

cmake --build build
cmake --install build
"""

platforms = supported_platforms()
# Filter to platforms with Vulkan support
filter!(p -> Sys.islinux(p) || Sys.isapple(p), platforms)
# Exclude FreeBSD (no Vulkan ICD)
filter!(p -> !Sys.isfreebsd(p), platforms)

products = [
    LibraryProduct("libvksift_jl", :libvksift_jl),
]

dependencies = [
    BuildDependency("Vulkan_Headers_jll"),
    Dependency("Xorg_libX11_jll"; platforms=filter(Sys.islinux, platforms)),
    BuildDependency("Xorg_xorgproto_jll"; platforms=filter(Sys.islinux, platforms)),
]

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat="1.6",
               preferred_gcc_version=v"9")
