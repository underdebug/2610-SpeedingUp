// MIT License

// Copyright (c) 2026 Mingjie Zhang (mingjie2026@gmail.com)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

// Read local quoted header references from a C++ source file.
function discoverHeaders(source) {
    const code = fs.readFileSync(source, 'utf8');
    const regex = /^\s*#\s*include\s*"([^"]+)"/gm;
    const headers = [];
    let match;

    while ((match = regex.exec(code)) !== null) {
        headers.push(match[1]);
    }

    return headers;
}

// Find a source file corresponding to a local header.
function findSource(header, sourceDir) {
    const base = header.replace(/\.(h|hpp|hh)$/i, '');
    for (const extension of ['.cu', '.cpp', '.cc', '.cxx']) {
        const source = path.resolve(sourceDir, `${base}${extension}`);
        if (fs.existsSync(source)) return source;
    }

    return null;
}

// Recursively collect source files referenced by local headers.
function collectSources(source, visited = new Set()) {
    source = path.resolve(source);

    if (visited.has(source)) {
        return visited;
    }

    visited.add(source);

    const sourceDir = path.dirname(source);
    const headers = discoverHeaders(source);

    for (const header of headers) {
        const cpp = findSource(header, sourceDir);

        if (cpp) {
            collectSources(cpp, visited);
        }
    }

    return visited;
}

// Compile one CUDA or C++ source file into an object file.
function compileSource(source, buildDir) {
    const object = path.join(buildDir, `${path.basename(source, path.extname(source))}.o`);

    execFileSync(process.env.NVCC || 'nvcc', [
        '-g',
        '-G',
        '-c',
        source,
        '-o',
        object
    ], { stdio: 'inherit' });

    return object;
}

// Link object files into the executable.
function linkObjects(objects, output) {
    execFileSync(process.env.NVCC || 'nvcc', [
        ...objects,
        '-o',
        output
    ], { stdio: 'inherit' });
}

function prepareGpu(entry, buildDir) {
    entry = path.resolve(entry);
    const programName = path.basename(entry).split('.')[0];
    buildDir = path.resolve(buildDir);
    const objectDir = path.join(buildDir, 'cuda-gdb');

    fs.mkdirSync(objectDir, { recursive: true });

    const sources = [...collectSources(entry)];
    const objects = sources.map(source => compileSource(source, objectDir));
    const output = path.join(buildDir, programName);

    linkObjects(objects, output);

    return output;
}

module.exports = { prepareGpu };
