const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
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
    // #version ... => #include "shader.h"
    text = text.replace(/^[ \t]*#version\b[^\r\n]*/, '#include "shader.h"');

    // main() => shader()
    text = text.replace(/\bmain\s*\(\s*\)/g, 'shader()');
    return text;
}

function replace_kernel(text){
    // Use the CPU definitions for CUDA headers in generated files.
    text = text.replace(/^[ \t]*#[ \t]*include[ \t]*[<"](?:cuda_[^>"\r\n]*|device_launch_parameters\.h)[>"]/gm, '#include "kernel.h"');
    return text;
}

function replace(text) {
    // check shader file
    text = replace_cpp(text);
    text = replace_shader(text);
    text = replace_kernel(text);
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
function generateCpuFile(source, buildDir, generated = new Map()) {
    source = path.resolve(source);
    if (generated.has(source)) return generated.get(source);

    const id = crypto.createHash('sha256').update(source).digest('hex').slice(0, 16);
    const output = path.join(buildDir, `${id}-${path.basename(source)}`);
    generated.set(source, output);

    let code = fs.readFileSync(source, 'utf8');
    code = code.replace(/^(\s*#\s*include\s*)"([^"\r\n]+)"/gm, (match, prefix, header) => {
        const local = path.resolve(path.dirname(source), header);
        if (!fs.existsSync(local) || !fs.statSync(local).isFile()) return match;
        return prefix + JSON.stringify(generateCpuFile(local, buildDir, generated));
    });
    code = replace_kernel(code);
    fs.writeFileSync(output, `#line 1 ${JSON.stringify(source)}\n${code}`);
    return output;
}

// Compile one C++ source file into an object file.
function compileSource(source, buildDir) {
    const object = path.join(buildDir, `${path.basename(source, path.extname(source))}.o`);

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
    fs.mkdirSync(objectDir, { recursive: true });

    const output = path.join(buildDir, programName);
    const sources = [...collectSources(entry)];
    const generated = new Map();
    const objects = sources.map(source => compileSource(generateCpuFile(source, objectDir, generated), objectDir));
        
    linkObjects(objects, output);

    return output;
}

module.exports = { prepareCpu };
