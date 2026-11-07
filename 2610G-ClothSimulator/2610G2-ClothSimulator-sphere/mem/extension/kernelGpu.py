"""Build a CUDA test and its referenced implementations as separate cached objects."""
from pathlib import Path
import argparse
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys


IMPLEMENTATIONS = ('.cu', '.cpp', '.cc', '.cxx', '.c')
HEADERS = ('.h', '.hpp', '.hxx', '.cuh')


def project_root(source):
    if os.environ.get('PROJECT_PATH'):
        return Path(os.environ['PROJECT_PATH']).resolve()
    return next((p for p in source.resolve().parents
                 if any((p / name).exists() for name in
                        ('.vscode', '.git', 'CMakeLists.txt', 'mem/extension'))),
                source.resolve().parent)


def original_source(source):
    # Older F4 builds left foo.cu_.cu.cu (and sometimes more .cu suffixes).
    source = source.resolve(strict=True)
    while source.name.endswith('.cu.cu') and source.with_suffix('').is_file():
        source = source.with_suffix('')
    return source


def build_directory(source):
    return project_root(source) / 'build' / 'cuda-gdb'


def executable_path(source):
    return project_root(source) / 'build' / source.name.split('.')[0]


def nvcc_command():
    if os.environ.get('NVCC'):
        return shlex.split(os.environ['NVCC'])
    compiler = shutil.which('nvcc')
    if not compiler:
        toolkit = Path(os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
                       or '/usr/local/cuda')
        compiler = str(toolkit / 'bin/nvcc')
    if not Path(compiler).is_file():
        raise RuntimeError('NVCC is required; set NVCC or CUDA_HOME')
    return [compiler]


def mask_comments(text):
    # Preserve directives and newlines, but ignore commented-out includes.
    return re.sub(r'R"([^ ()\\\t\r\n]{0,16})\(.*?\)\1"|"(?:\\.|[^"\\])*"|'
                  r"'(?:\\.|[^'\\])*'|/\*.*?\*/|//[^\n]*",
                  lambda m: re.sub(r'[^\n]', ' ', m[0])
                  if m[0].startswith(('//', '/*', 'R"')) else m[0],
                  text, flags=re.DOTALL)


class Dependencies:
    def __init__(self, source, directories):
        self.source = source
        self.directories = directories
        self.files = set()
        self.sources = {source}
        self.embedded = set()
        self.text = []

    def resolve(self, name, parent):
        return next((p.resolve() for p in
                     [parent / name, *(d / name for d in self.directories)]
                     if p.is_file()), None)

    def visit(self, filename):
        if filename in self.files:
            return
        self.files.add(filename)
        text = mask_comments(filename.read_text(encoding='utf-8'))
        self.text.append(text)
        for name in re.findall(r'^\s*#\s*include\s*[<"]([^>"\n]+)[>"]', text, re.M):
            dependency = self.resolve(name, filename.parent)
            if dependency is None:
                continue  # System headers are tracked by NVCC's depfiles.
            if dependency.suffix in IMPLEMENTATIONS:
                self.embedded.add(dependency)
            self.visit(dependency)
        if filename.suffix in HEADERS:
            for suffix in IMPLEMENTATIONS:
                companion = self.resolve(filename.stem + suffix, filename.parent)
                if companion:
                    self.sources.add(companion)
                    self.visit(companion)
                    break

    def collect(self):
        self.visit(self.source)
        # Legacy tests may still include an implementation directly. Never link
        # that implementation a second time as an independent object.
        return sorted(self.sources - self.embedded)


def architecture_flags(root):
    architectures = os.environ.get('CUDAARCHS', '')
    settings = root / '.vscode/settings.json'
    if not architectures and settings.is_file():
        match = re.search(r'"CMAKE_CUDA_ARCHITECTURES"\s*:\s*"([\d;\-a-z]+)"',
                          mask_comments(settings.read_text()))
        if match:
            architectures = match[1]
    flags = []
    for arch in filter(None, architectures.split(';')):
        match = re.fullmatch(r'(\d+[af]?)(?:-(real|virtual))?', arch)
        if not match:
            raise ValueError('CUDAARCHS must contain numeric architectures, e.g. 89 or 80;89')
        number, kind = match.groups()
        code = (f'sm_{number}' if kind == 'real' else f'compute_{number}'
                if kind == 'virtual' else f'[compute_{number},sm_{number}]')
        flags.append(f'--generate-code=arch=compute_{number},code={code}')
    return flags


def ninja_path(path):
    text = str(path)
    if '\n' in text or '\r' in text:
        raise ValueError('Build paths must not contain newlines')
    return text.replace('$', '$$').replace(' ', '$ ').replace(':', '$:')


def ninja_command(args):
    return shlex.join(str(a) for a in args).replace('$', '$$')


def write_if_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def build(source, executable=None, coder=None):
    source = original_source(source)
    root = project_root(source)
    folder = build_directory(source)
    executable = (executable or executable_path(source)).resolve()
    directories = [source.parent, root, root / 'src', root / 'inc', root / 'include']
    extension = Path(os.environ.get('EXTENSION_PATH', root / 'mem/extension')).resolve()
    directories.append(extension)
    directories = list(dict.fromkeys(d for d in directories if d.is_dir()))
    dependencies = Dependencies(source, directories)
    sources = dependencies.collect()
    program = '\n'.join(dependencies.text)
    # Keep CLI and editor commands identical when no extension header is used.
    includes = [flag for d in directories
                if d != extension or any(p.is_relative_to(extension) for p in dependencies.files)
                for flag in ('-I', str(d))]
    libraries = []
    for header, package in ((r'json/json\.h', 'jsoncpp'),
                            (r'Eigen/[^>"\n]+', 'eigen3'),
                            (r'GLFW/glfw3\.h', 'glfw3')):
        if re.search(r'#\s*include\s*[<"]' + header + r'[>"]', program):
            includes.extend(shlex.split(subprocess.check_output(
                ['pkg-config', '--cflags', package], text=True)))
            libraries.extend(shlex.split(subprocess.check_output(
                ['pkg-config', '--libs', package], text=True)))
    if re.search(r'#\s*include\s*[<"]glad/glad\.h[>"]', program):
        glad = root / 'lib/glad'
        if (glad / 'src/glad.c').is_file():
            includes.extend(['-I', str(glad / 'include')])
            sources.append(glad / 'src/glad.c')
            libraries.append('-ldl')
    for header, library in (('cublas_v2.h', 'cublas'), ('cusparse.h', 'cusparse')):
        if re.search(r'#\s*include\s*[<"]' + re.escape(header) + r'[>"]', program):
            libraries.append('-l' + library)
    if executable in dependencies.files:
        raise ValueError('Executable must not overwrite a source file')
    compiler = nvcc_command()
    options = ['-std=c++17', '-g', '-G', '-O0', *architecture_flags(root),
               '--diag-suppress=20015,177,20011,20012,20013,20014,940',
               *shlex.split(os.environ.get('NVCCFLAGS', ''))]

    # Single-file projects should build directly instead of generating a
    # per-implementation ninja graph. This keeps the debug flow working when a
    # source file is intentionally self-contained.
    single_file = os.environ.get('MEM_CUDA_SINGLE_FILE', '').lower() in {'1', 'true', 'yes', 'on'}
    if single_file or len(sources) == 1:
        command = [*compiler, *options, *includes, '-x', 'cu', str(source), '-o', str(executable), *libraries]
        print(f'CUDA single-file build: {source}', flush=True)
        return subprocess.run(command).returncode

    # Preserve cached objects when upgrading from build/<name>/<name>.
    legacy = executable_path(source)
    if legacy.is_dir() and (legacy / 'build.ninja').is_file() and not folder.exists():
        legacy.rename(folder)
        old_executable = folder / source.name.split('.')[0]
        if old_executable.is_file():
            old_executable.rename(legacy)
    folder.mkdir(parents=True, exist_ok=True)
    executable.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Generated by kernelGpu.py. One object per referenced implementation.',
        'rule compile',
        '  command = ' + ninja_command([*compiler, *options, *includes])
        + ' -MD -MT $out -MF $out.d -x cu -dc $in -o $out',
        '  depfile = $out.d',
        '  deps = gcc',
        '  description = CUDA $out',
        'rule link',
        '  command = ' + ninja_command([*compiler, *options])
        + ' $in -o $out ' + ninja_command(libraries),
        '  description = LINK $out',
    ]
    objects = []
    for filename in sorted(set(sources)):
        name = filename.name
        if sum(p.name == name for p in sources) > 1:
            name = hashlib.sha256(str(filename).encode()).hexdigest()[:12] + '-' + name
        obj = name + '.o'
        objects.append(ninja_path(obj))
        lines.append(f'build {ninja_path(obj)}: compile {ninja_path(filename)}')
    lines.extend([f'build {ninja_path(executable)}: link ' + ' '.join(objects),
                  f'default {ninja_path(executable)}', ''])
    write_if_changed(folder / 'build.ninja', '\n'.join(lines))
    print(f'CUDA build: {folder} ({len(objects)} separate objects)', flush=True)
    ninja = shutil.which('ninja')
    if not ninja:
        raise RuntimeError('Ninja is required for incremental CUDA builds')
    return subprocess.run([ninja, '-C', str(folder), '-j',
                           os.environ.get('MEM_CUDA_JOBS', '2')]).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    # Accept the old extension's positional output argument during upgrades.
    parser.add_argument('coder', nargs='?', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    source = original_source(args.source)
    executable = (args.output or executable_path(source)).resolve()
    result = build(source, executable)
    if result or not args.debug:
        return result
    debugger = shutil.which('cuda-gdb')
    if not debugger:
        raise RuntimeError('CUDA-GDB is required for --debug')
    return subprocess.run([debugger, '-q', '--args', str(executable)], cwd=source.parent).returncode


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f'kernelGpu.py: {error}', file=sys.stderr)
        sys.exit(1)
