import os
import numpy as np
import pybind11
import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext


def get_include_dirs():
    include_dirs = [
        pybind11.get_include(),
        np.get_include(),
    ]

    # compatibility when run in python_bindings
    bindings_dir = "python"
    if bindings_dir in os.path.basename(os.getcwd()):
        include_dirs.extend(["../", "../third_party/helpa"])
    else:
        include_dirs.extend(["./", "./third_party/helpa"])

    return include_dirs


def get_source_files():
    bindings_dir = "python"
    if bindings_dir in os.path.basename(os.getcwd()):
        return ["./bindings.cc"]
    else:
        return ["./python/bindings.cc"]


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    if has_flag(compiler, "-std=c++20"):
        return "-std=c++20"
    else:
        raise RuntimeError(
            "Unsupported compiler -- at least C++20 support " "is needed!"
        )


def _omp_flags(compiler, system):
    """Return (compile_flags, link_flags) for OpenMP, portably.

    Linux gcc/clang ship libgomp/libomp (-fopenmp just works). Apple clang needs
    Homebrew's libomp; if it isn't present we build single-threaded (the
    ``#pragma omp`` directives are simply ignored) rather than failing the build.
    """
    if system == "Darwin":
        for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
            if os.path.isdir(prefix):
                return (
                    ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
                    [f"-L{prefix}/lib", "-lomp"],
                )
        print("glass build: libomp not found — building single-threaded "
              "(`brew install libomp` for parallelism)")
        return ([], [])
    if has_flag(compiler, "-fopenmp"):
        return (["-fopenmp"], ["-fopenmp"])
    return ([], [])


class BuildExt(build_ext):
    """Handle compiler flags per platform.

    The previous build hardcoded ``-march=native -lrt -fopenmp``, which only works
    on x86 Linux: ``-lrt`` isn't a macOS library, Apple clang needs libomp rather
    than plain ``-fopenmp``, and ``-march=native`` isn't valid for Apple-clang
    arm64. This keeps the existing x86 behaviour (``-march=native`` by default) and
    makes the build work on macOS (Apple Silicon and Intel) and on aarch64:

    * x86_64 → ``-march=native`` (unchanged). Override with ``GLASS_MARCH`` (e.g. a
      fixed ``-mavx2`` baseline when building portable/redistributable wheels).
    * aarch64/arm64 → no ``-march`` flag; NEON is implicit (``__aarch64__``), and
      Apple clang rejects ``-march=native`` here.
    * ``-lrt`` only on Linux; OpenMP via libomp on macOS when present.
    """

    def build_extensions(self):
        import platform

        system = platform.system()
        mach = platform.machine().lower()

        opts = ["-Ofast", "-fPIC", "-ftree-vectorize"]
        opts.append(f'-DVERSION_INFO="{self.distribution.get_version()}"')
        opts.append(cpp_flag(self.compiler))
        if has_flag(self.compiler, "-fvisibility=hidden"):
            opts.append("-fvisibility=hidden")

        march = os.environ.get("GLASS_MARCH")
        if march:
            # Split so a multi-flag value (e.g. "-mavx2 -mfma -mf16c") reaches the
            # compiler as separate tokens rather than one unrecognized argument.
            opts.extend(march.split())
        elif mach in ("x86_64", "amd64"):
            opts.append("-march=native")
        # arm64/aarch64: NEON is implicit (no -march needed).

        link = ["-pthread"]
        if system == "Linux":
            link.append("-lrt")
        omp_c, omp_l = _omp_flags(self.compiler, system)
        opts += omp_c
        link += omp_l

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(link)

        build_ext.build_extensions(self)


def create_extension(version_info):
    libraries = []
    extra_objects = []

    ext_modules = [
        Extension(
            "glass",
            get_source_files(),
            include_dirs=get_include_dirs(),
            libraries=libraries,
            language="c++",
            extra_objects=extra_objects,
        ),
    ]

    return ext_modules


def get_build_ext_cmdclass():
    return {"build_ext": BuildExt}
