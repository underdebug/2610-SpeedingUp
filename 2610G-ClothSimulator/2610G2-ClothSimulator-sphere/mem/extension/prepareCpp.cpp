// Build support for isolated, stateless C++ method tests.
const fs = require('fs');
const path = require('path');

const directive = /^[ \t]*#pragma[ \t]+mem_methods[ \t]+"([^"\n]+)"[ \t]+(\w+)[ \t]+([\w \t]+)$/gm;

function discoverMethods(text, source) {
    if (!source || !/\.cu_\.(?:cc|cpp|cxx)$/.test(source)) return [];
    const filename = source.replace(/_\.(?:cc|cpp|cxx)$/, '');
    if (!fs.existsSync(filename)) return [];
    const className = path.basename(filename, '.cu');
    const clean = mask(text);
    const objects = [...clean.matchAll(new RegExp(`\\b${className}\\s+(\\w+)\\s*(?:;|[={])`, 'g'))];
    const names = new Set();
    for (const object of objects) {
        for (const call of clean.matchAll(new RegExp(`\\b${object[1]}\\s*\\.\\s*(\\w+)\\s*\\(`, 'g'))) {
            names.add(call[1]);
        }
    }
    return names.size ? [{ filename, className, names: [...names] }] : [];
}

// Preserve offsets and line numbers while ignoring comments and quoted literals.
function mask(text) {
    return text.replace(/\/\*[\s\S]*?\*\/|\/\/[^\r\n]*|R"([^ ()\\\t\r\n]{0,16})\([\s\S]*?\)\1"|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g,
        value => value.replace(/[^\r\n]/g, ' '));
}

function extractMethod(text, filename, className, name) {
    const clean = mask(text);
    const pattern = new RegExp(`^[ \\t]*([^;{}\\n]+\\b${className}::${name}\\s*\\([^;{}]*?\\)\\s*(?:const\\s*)?(?:noexcept\\s*)?)\\{`, 'gm');
    const matches = [...clean.matchAll(pattern)];
    if (matches.length !== 1) throw new Error(`Expected one definition of ${className}::${name} in ${filename}; found ${matches.length}`);
    const match = matches[0];
    const start = match.index;
    const open = start + match[0].lastIndexOf('{');
    let depth = 1, end = open + 1;
    for (; end < clean.length && depth; ++end) {
        if (clean[end] === '{') ++depth;
        if (clean[end] === '}') --depth;
    }
    if (depth) throw new Error(`Unclosed body for ${className}::${name}`);
    const signature = text.slice(start, open).trim();
    const line = text.slice(0, start).split('\n').length;
    return {
        declaration: signature.replace(`${className}::`, '') + ';',
        body: clean.slice(open + 1, end - 1),
        definition: `#line ${line} ${JSON.stringify(filename)}\n${text.slice(start, end)}\n`
    };
}

function renderMethods(filename, className, names, automatic = false) {
    const implementation = fs.readFileSync(filename, 'utf8');
    const available = new Set([...mask(implementation).matchAll(
        new RegExp(`\\b${className}::(\\w+)\\s*\\(`, 'g'))].map(match => match[1]));
    const methods = new Map();
    const pending = [...names];
    while (pending.length) {
        const name = pending.shift();
        if (methods.has(name)) continue;
        const method = extractMethod(implementation, filename, className, name);
        methods.set(name, method);
        if (automatic) {
            // Follow unqualified member calls, including this->method().
            for (const call of method.body.matchAll(/(?:\bthis\s*->\s*|(?<![\w.:>]))\b(\w+)\s*\(/g)) {
                if (available.has(call[1]) && !methods.has(call[1])) pending.push(call[1]);
            }
        }
    }
    return `class ${className} {\npublic:\n${[...methods.values()].map(m => m.declaration).join('\n')}\n};\n`
        + [...methods.values()].map(m => m.definition).join('\n');
}

// Paired .cu_.cc tests can discover stateless methods automatically. The caller
// still supplies type headers; stateful methods and overloads are unsupported.
async function prepareCpp(source, buildDirectory) {
    const text = await fs.promises.readFile(source, 'utf8');
    const explicit = new RegExp(directive.source, 'm').test(text);
    const groups = explicit ? [] : discoverMethods(text, source);
    if (!explicit && groups.length === 0) return { source, args: [] };
    let generated = text.replace(directive, (line, relative, className, names, offset) => {
        const filename = path.resolve(path.dirname(source), relative);
        return renderMethods(filename, className, names.trim().split(/\s+/))
            + `#line ${text.slice(0, offset + line.length).split('\n').length} ${JSON.stringify(source)}\n`;
    });
    if (!explicit) {
        const main = /\bint\s+main\s*\(/.exec(mask(text));
        if (!main) throw new Error('Automatic method tests require int main()');
        const insertion = text.lastIndexOf('\n', main.index) + 1;
        const definitions = groups.map(group =>
            renderMethods(group.filename, group.className, group.names, true)).join('\n');
        generated = text.slice(0, insertion) + definitions
            + `#line ${text.slice(0, insertion).split('\n').length} ${JSON.stringify(source)}\n`
            + text.slice(insertion);
    }
    await fs.promises.mkdir(buildDirectory, { recursive: true });
    const generatedSource = path.join(buildDirectory, path.basename(source) + '.generated.cpp');
    await fs.promises.writeFile(generatedSource, `#line 1 ${JSON.stringify(source)}\n${generated}`);
    const cudaRoot = process.env.CUDA_PATH || process.env.CUDA_HOME || '/usr/local/cuda';
    return { source: generatedSource, args: ['-I', path.dirname(source), '-I', path.join(cudaRoot, 'include')] };
}

module.exports = { prepareCpp };
