declare const workerScriptUrl: string;

const worker = new Worker(workerScriptUrl);

worker.onmessage = function ({ data }) {
  document.body.append(data);
};
