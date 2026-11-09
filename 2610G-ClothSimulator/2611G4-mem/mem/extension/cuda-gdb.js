const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

function replace_shader(text){
    // #version ... --> #include "shader.h"
    text = text.replace(/^[ \t]*#version\b[^\r\n]*/, '#include "shader.h"');

    // main() --> shader()
    text = text.replace(/\bmain\s*\(\s*\)/g, 'shader()');
    return text;
}

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

function replace(text) {
    // check shader file
    if (/^[ \t]*#version\b/.test(text)) {
        text = replace_shader(text);
    }

    text = replace_cpp(text);
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
