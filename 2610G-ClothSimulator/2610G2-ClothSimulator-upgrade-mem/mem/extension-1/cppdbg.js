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

function replace_cpp(text){
    //-------------------------------------------------------------------------------------
    // Input:                   Output:
    // int x; // hello          int x; ········
    // run(); // test           run(); ·······
    // // comment               ··········
    text = text.replace(/\/\/[^\r\n]*/g, value => ' '.repeat(value.length));
    
    //-------------------------------------------------------------------------------------
    // Input:                   Output:
    // a /* note */ + b         a ·········· + b
    // a /* note                a ·······
    //    more */ + b               ·········· + b
    text = text.replace(/\/\*[\s\S]*?\*\//g, value => value.replace(/[^\r\n]/g, ' '));
    return text;
}

function replace_shader(text){
    if (!/^[ \t]*#version\b/m.test(text)){
        return text;
    } 

    // #version ... => #include "shader.h"
    text = text.replace(/^[ \t]*#version\b[^\r\n]*/, '#include "shader.h"');

    // main() => shader()
    text = text.replace(/\bmain\s*\(\s*\)/g, 'shader()');
    return text;
}

function replace_kernel(text){
    // #include <cuda_...> => #include "kernel.h"
    text = text.replace(/^[ \t]*#[ \t]*include[ \t]*[<"]cuda_[^>"\r\n]*[>"]/gm, '#include "kernel.h"');
 
    // #include <device_launch_parameters.h> => #include "kernel.h"
    text = text.replace(/^[ \t]*#[ \t]*include[ \t]*[<"]device_launch_parameters\.h[>"]/gm, '#include "kernel.h"');
    return text;
}

function replace_kernel_func(text){
    // kernel<<<grid, block>>>(args) => run_kernel(dim3(grid), dim3(block), ...)
    // kernel<<<grid, block>>>( => kernel name and launch configuration
    const launch = /\b([A-Za-z_]\w*(?:::\w+)*)[ \t]*<<<([^\r\n]*?)>>>[ \t]*\(/g;
    // Matches => replacement ranges and generated calls
    const replacements = [];
    let match;

    while ((match = launch.exec(text)) !== null) {
        const configStart = match.index + match[0].indexOf('<<<') + 3;
        const config = match[2];
        let depth = 0;
        let comma = -1;
        for (let i = 0; i < config.length; i++) {
            if ('([{'.includes(config[i])) depth++;
            else if (')]}'.includes(config[i])) depth--;
            else if (config[i] === ',' && depth === 0) {
                if (comma !== -1) throw new Error('CPU replay supports only grid and block launch dimensions');
                comma = i;
            }
        }
        if (comma < 0) throw new Error('Expected grid and block dimensions in CUDA launch');

        const argsStart = launch.lastIndex;
        let end = argsStart;
        depth = 1;
        while (end < text.length && !/[\r\n]/.test(text[end]) && depth) {
            if (text[end] === '(') depth++;
            else if (text[end] === ')') depth--;
            end++;
        }
        if (depth) continue;

        const grid = text.slice(configStart, configStart + comma);
        const block = text.slice(configStart + comma + 1, configStart + config.length);
        const args = text.slice(argsStart, end - 1);
        replacements.push([match.index, end, `run_kernel(dim3(${grid}), dim3(${block}), [&]() { ${match[1]}(${args}); })`]);
        launch.lastIndex = end;
    }
    for (const [start, end, replacement] of replacements.reverse()) {
        text = text.slice(0, start) + replacement + text.slice(end);
    }
    return text;
}

function replace(text) {
    text = replace_cpp(text);
    text = replace_shader(text);
    text = replace_kernel(text);
    text = replace_kernel_func(text);
    return text;
}

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

// Generate CPU copies of sources and their local headers without changing the originals.
function generateCpuFile(source, buildDir, sourceRoot, generated = new Map()) {
    source = path.resolve(source);
    if (generated.has(source)) return generated.get(source);

    let relative = path.relative(sourceRoot, source);
    if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
        relative = path.join('__external__', source.replace(/^([A-Za-z]:)?[/\\]+/, ''));
    }
    const destination = path.join(buildDir, relative);
    generated.set(source, destination);

    let code = fs.readFileSync(source, 'utf8');
    code = code.replace(/^(\s*#\s*include\s*)"([^"\r\n]+)"/gm, (match, prefix, header) => {
        const local = path.resolve(path.dirname(source), header);
        if (!fs.existsSync(local) || !fs.statSync(local).isFile()) return match;
        return prefix + JSON.stringify(generateCpuFile(local, buildDir, sourceRoot, generated));
    });

    code = replace(code);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.writeFileSync(destination, `#line 1 ${JSON.stringify(source)}\n${code}`);
    return destination;
}

// Compile one C++ source file into an object file.
function compileSource(source, buildDir, generatedDir) {
    const object = path.join(buildDir, `${path.relative(generatedDir, source)}.o`);
    fs.mkdirSync(path.dirname(object), { recursive: true });

    execFileSync(process.env.CXX || 'g++', [
        '-std=c++17',
        '-g',
        '-x', 'c++', // compile .cu file like .cpp NODELETE
        '-I', __dirname,
        '-c',
        source,
        '-o',
        object
    ], { encoding: 'utf8' });

    return object;
}

// Link object files into the executable.
function linkObjects(objects, output) {
    execFileSync(process.env.CXX || 'g++', [
        ...objects,
        '-o',
        output
    ], { encoding: 'utf8' });
}

function prepareCpu(entry, buildDir) {
    entry = path.resolve(entry);
    const programName = path.basename(entry).split('.')[0];
    buildDir = path.resolve(buildDir);
    
    const objectDir = path.join(buildDir, 'cppdbg');
    const generatedDir = path.join(buildDir, 'generated');
    fs.mkdirSync(objectDir, { recursive: true });

    const output = path.join(buildDir, programName);
    const sources = [...collectSources(entry)];
    const generated = new Map();
    const objects = sources.map(source =>
        compileSource(generateCpuFile(source, generatedDir, path.dirname(entry), generated), objectDir, generatedDir)
    );
        
    linkObjects(objects, output);

    return output;
}

module.exports = { prepareCpu };
