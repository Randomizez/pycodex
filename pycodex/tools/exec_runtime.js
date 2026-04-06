const readline = require('readline');
const { stdin, stdout } = require('process');

const pending = new Map();
let storedValues = {};
let toolMap = {};
let initialized = false;

function send(message) {
  stdout.write(JSON.stringify(message) + '\n');
}

function stringifyValue(value) {
  if (typeof value === 'string') {
    return value;
  }
  if (value === undefined) {
    return 'undefined';
  }
  try {
    return JSON.stringify(value);
  } catch (_error) {
    return String(value);
  }
}

function text(value) {
  send({ type: 'output_text', text: stringifyValue(value) });
}

function notify(value) {
  text(value);
}

function image(value) {
  if (typeof value === 'string') {
    send({ type: 'output_image', image_url: value, detail: null });
    return;
  }
  if (value && typeof value === 'object' && typeof value.image_url === 'string') {
    send({
      type: 'output_image',
      image_url: value.image_url,
      detail: value.detail === undefined ? null : value.detail,
    });
    return;
  }
  throw new Error('image(...) expects an image URL string or { image_url, detail? }');
}

function store(key, value) {
  storedValues[key] = value;
}

function load(key) {
  return storedValues[key];
}

function exit() {
  const error = new Error('__CODEX_EXIT__');
  error.__codexExit = true;
  throw error;
}

async function yield_control() {
  send({ type: 'yield' });
}

function createToolCaller(toolName) {
  return function callTool(argumentsValue = {}) {
    const id = `${toolName}_${Math.random().toString(16).slice(2)}`;
    send({
      type: 'tool_call',
      id,
      tool_name: toolName,
      arguments: argumentsValue,
    });
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
    });
  };
}

function initialize(message) {
  storedValues = message.stored_values || {};
  toolMap = {};
  for (const tool of message.tools || []) {
    toolMap[tool.js_name] = createToolCaller(tool.tool_name);
  }
  const allTools = Object.freeze(
    (message.tools || []).map((tool) => ({
      name: tool.js_name,
      description: tool.description,
    }))
  );

  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  const fn = new AsyncFunction(
    'tools',
    'text',
    'image',
    'store',
    'load',
    'exit',
    'notify',
    'ALL_TOOLS',
    'yield_control',
    'console',
    message.source
  );

  (async () => {
    try {
      await fn(
        toolMap,
        text,
        image,
        store,
        load,
        exit,
        notify,
        allTools,
        yield_control,
        undefined
      );
      send({ type: 'result', stored_values: storedValues, error_text: null });
    } catch (error) {
      if (error && error.__codexExit) {
        send({ type: 'result', stored_values: storedValues, error_text: null });
        return;
      }
      const errorText = error && error.stack ? error.stack : String(error);
      send({ type: 'result', stored_values: storedValues, error_text: errorText });
    }
  })();
}

const rl = readline.createInterface({ input: stdin, crlfDelay: Infinity });
rl.on('line', (line) => {
  if (!line.trim()) {
    return;
  }
  const message = JSON.parse(line);
  if (message.type === 'init' && !initialized) {
    initialized = true;
    initialize(message);
    return;
  }
  if (message.type === 'tool_result') {
    const entry = pending.get(message.id);
    if (!entry) {
      return;
    }
    pending.delete(message.id);
    if (message.ok) {
      entry.resolve(message.result);
    } else {
      entry.reject(new Error(message.error || 'nested tool failed'));
    }
  }
});
