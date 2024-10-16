import { Bitmap, BlendMode, Jimp } from "jimp";
import { Tensor } from "onnxruntime-web";

export async function getImageTensorFromPath(path: string, dims: number[] = [1, 3, 1024, 1024]) {
  const image = (await loadImageFromPath(path)).bitmap;
  const imageTensor = imageDataToTensor(image,[dims[2],dims[3]]);
  return imageTensor;
}
async function loadImageFromPath(path: string) {
  return await Jimp.read(path);
}


function imageDataToTensor(image: Bitmap, modelInputSize: [number, number]) {
  const imageBufferData = image.data;
  const height = image.height;
  const width = image.width;

  const [redArray, greenArray, blueArray]: [number[], number[], number[]] = [[], [], []];
  
  for (let i = 0; i < imageBufferData.length; i += 4) {
    redArray.push(imageBufferData[i]);
    greenArray.push(imageBufferData[i + 1]);
    blueArray.push(imageBufferData[i + 2]);
  }

  const transposedData = redArray.concat(greenArray).concat(blueArray);

  const float32Data = new Float32Array(3 * height * width);
  for (let i = 0; i < transposedData.length; i++) {
    float32Data[i] = transposedData[i] / 255.0;
  }

  const reshapedTensor = new Float32Array(3 * height * width);
  
  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      reshapedTensor[h * width + w] = float32Data[h * width + w];                      
      reshapedTensor[height * width + h * width + w] = float32Data[height * width + h * width + w]; 
      reshapedTensor[2 * height * width + h * width + w] = float32Data[2 * height * width + h * width + w];
    }
  }

  const mean = [0.5, 0.5, 0.5];
  const std = [1.0, 1.0, 1.0];

  for (let i = 0; i < height * width; i++) {
    reshapedTensor[i] = (reshapedTensor[i] - mean[0]) / std[0];             // R
    reshapedTensor[height * width + i] = (reshapedTensor[height * width + i] - mean[1]) / std[1]; // G
    reshapedTensor[2 * height * width + i] = (reshapedTensor[2 * height * width + i] - mean[2]) / std[2]; // B
  }

  const resizedTensor = resizeImageTensor(reshapedTensor, height, width, modelInputSize);

  return resizedTensor;
}

function resizeImageTensor(tensor: Float32Array, originalHeight: number, originalWidth: number, modelInputSize: [number, number]): Float32Array {
  const [newHeight, newWidth] = modelInputSize;
  
  const resizedTensor = new Float32Array(3 * newHeight * newWidth);
  
  for (let c = 0; c < 3; c++) {
    for (let h = 0; h < newHeight; h++) {
      for (let w = 0; w < newWidth; w++) {
        const oldH = Math.floor((h / newHeight) * originalHeight);
        const oldW = Math.floor((w / newWidth) * originalWidth);
        resizedTensor[c * newHeight * newWidth + h * newWidth + w] = tensor[c * originalHeight * originalWidth + oldH * originalWidth + oldW];
      }
    }
  }
  
  return resizedTensor;
}

export function postprocessImage(result: Tensor, imSize: number[]): ImageData {
  console.log("Result Dims: ", result.dims);
  const [height, width] = imSize;
  const resultData = result.data as Float32Array;
  const [channels, resultHeight, resultWidth] = result.dims.slice(1);

  const resizedData = resizeBilinear(resultData, [1, resultHeight, resultWidth], imSize);
console.log(channels)
let maxVal = -Infinity;
let minVal = Infinity;

for (let i = 0; i < resizedData.length; i++) {
    if (resizedData[i] > maxVal) maxVal = resizedData[i];
    if (resizedData[i] < minVal) minVal = resizedData[i];
}

    const normalizedData = new Float32Array(resizedData.length);
    for (let i = 0; i < resizedData.length; i++) {
        normalizedData[i] = (resizedData[i] - minVal) / (maxVal - minVal);
    }

    const uint8Data = new Uint8ClampedArray(normalizedData.length);
    for (let i = 0; i < normalizedData.length; i++) {
        uint8Data[i] = Math.round(normalizedData[i] * 255);
    }

  const imageData = new ImageData(width, height);
  let idx = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * channels;
      imageData.data[idx] = uint8Data[i];
      imageData.data[idx + 1] = uint8Data[i + 1];
      imageData.data[idx + 2] = uint8Data[i + 2];
      imageData.data[idx + 3] = 255; 
      idx += 4;
    }
  }

  return imageData;
}

function resizeBilinear(data: Float32Array, inputDims: number[], outputSize: number[]): Float32Array {
  const [channels, inHeight, inWidth] = inputDims;
  const [outHeight, outWidth] = outputSize;
  const output = new Float32Array(channels * outHeight * outWidth);

  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < outHeight; y++) {
      for (let x = 0; x < outWidth; x++) {
        const inX = (x / outWidth) * inWidth;
        const inY = (y / outHeight) * inHeight;

        const x0 = Math.floor(inX);
        const x1 = Math.min(x0 + 1, inWidth - 1);
        const y0 = Math.floor(inY);
        const y1 = Math.min(y0 + 1, inHeight - 1);

        const xWeight = inX - x0;
        const yWeight = inY - y0;

        const top = (1 - xWeight) * data[c * inHeight * inWidth + y0 * inWidth + x0] +
                    xWeight * data[c * inHeight * inWidth + y0 * inWidth + x1];
        const bottom = (1 - xWeight) * data[c * inHeight * inWidth + y1 * inWidth + x0] +
                       xWeight * data[c * inHeight * inWidth + y1 * inWidth + x1];

        output[c * outHeight * outWidth + y * outWidth + x] = (1 - yWeight) * top + yWeight * bottom;
      }
    }
  }

  return output;
}


export async function maskImage(originalImage: string, maskImage: string): Promise<Buffer> {
  try {
      const original = await Jimp.read(originalImage);
      const mask = await Jimp.read(maskImage);

      original.composite(mask, 0, 0, {
          mode: BlendMode.DARKEN,
          opacitySource: 1,
          opacityDest:1,

      });

      original.scan(0, 0, original.bitmap.width, original.bitmap.height, (x, y, idx) => {
        const red = original.bitmap.data[idx + 0];
        const green = original.bitmap.data[idx + 1];
        const blue = original.bitmap.data[idx + 2];
        const alpha = original.bitmap.data[idx + 3];
  
        if ((red <= 20 && green <= 20 && blue <= 20) && alpha === 255) {
          // Pikseli şeffaf yap
          original.bitmap.data[idx + 3] = 0;
        }
      });

      return original.bitmap.data
  } catch (error) {
      console.error("Error:", error);
      throw error; 
  }
}