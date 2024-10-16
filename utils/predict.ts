import { Tensor } from 'onnxruntime-web';
import { getImageTensorFromPath } from './imageHelper';
import { runSqueezenetModel } from './modelHelper';

export async function inferenceSqueezenet(image: HTMLImageElement): Promise<[Tensor,number]> {
  const batch_size = 1
  const MODEL_SHAPES = [batch_size, 3, 1024, 1024];

  const input = await getImageTensorFromPath(image.src);
  console.log(input)

  const tensor = new Tensor("float32", input, MODEL_SHAPES);
  console.log(tensor.dims)
  
  const [predictions, inferenceTime] = await runSqueezenetModel(tensor);
  return [predictions, inferenceTime];
}