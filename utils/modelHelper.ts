
import * as ort from 'onnxruntime-web';

export async function runSqueezenetModel(preprocessedData: ort.Tensor): Promise<[ort.Tensor, number]> {
  
  // Create session and set options. See the docs here for more options: 
  //https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#graphOptimizationLevel
  const session = await ort.InferenceSession
                          .create('./_next/static/chunks/pages/model.onnx', 
                          { executionProviders: ['wasm'] , graphOptimizationLevel: "all"});
  console.log('Inference session created')
  ort.env.wasm.numThreads = 0
  const availableBackends = ort.env.webgpu
    console.log(availableBackends)
  // Run inference and get results.
  const [results, inferenceTime] =  await runInference(session, preprocessedData);
  return [results, inferenceTime];
}

async function runInference(session: ort.InferenceSession, preprocessedData: ort.Tensor): Promise<[ort.Tensor, number]> {
  const start = new Date();
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;
  
  const outputData = await session.run(feeds);
  console.log(outputData)
  console.log(session.outputNames[0])

  const end = new Date();
  const inferenceTime = (end.getTime() - start.getTime())/1000;

  const output = outputData[session.outputNames[0]];
  return [output, inferenceTime];
}
