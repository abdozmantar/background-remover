/* eslint-disable @next/next/no-img-element */
/* eslint-disable @typescript-eslint/no-unused-vars */
"use client"

import React, { useState } from 'react';
import { inferenceSqueezenet } from '@/utils/predict';
import { maskImage, postprocessImage } from '@/utils/imageHelper';

const Home: React.FC = () => {
  const [outputSrc, setOutputSrc] = useState<string | null>(null);


  const runModel = async (imagePath: string) => {
    const image = new Image();
    image.src = imagePath;
    image.crossOrigin = 'Anonymous';

    image.onload = async () => {
      const [inferenceResult, inferenceTime] = await inferenceSqueezenet(image);

      console.log(inferenceResult);
      console.log(inferenceTime);

      const mask = postprocessImage(inferenceResult, [image.height, image.width]);

      const originalImageData = await getImageData(image);
      const maskCanvas = document.createElement('canvas');
      const maskContext = maskCanvas.getContext('2d');

      if (!maskContext) return;

      maskCanvas.width = image.width;
      maskCanvas.height = image.height;
      maskContext.putImageData(mask, 0, 0);

      try {
        const finalImage = await maskImage(originalImageData, maskCanvas.toDataURL());

        const outputCanvas = document.createElement('canvas');
        const outputContext = outputCanvas.getContext('2d');

        if (!outputContext) return;

        outputCanvas.width = image.width;
        outputCanvas.height = image.height;
        const imageData = new ImageData(new Uint8ClampedArray(finalImage), image.width, image.height);
        outputContext.putImageData(imageData, 0, 0);
        setOutputSrc(outputCanvas.toDataURL());

      } catch (error) {
        console.error("Error ", error);
      }
    };
  };

  async function getImageData(image: HTMLImageElement): Promise<string> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (ctx === null)
      throw new Error("Canvas context creation failed");

    canvas.width = image.width;
    canvas.height = image.height;

    ctx.drawImage(image, 0, 0);
    return canvas.toDataURL();
  }

  return (
    <div>
      <h1>ONNX Model Demo</h1>
      <button onClick={() => runModel("https://letsenhance.io/static/8f5e523ee6b2479e26ecc91b9c25261e/1015f/MainAfter.jpg")}>Launch</button>
      {outputSrc && <img src={outputSrc} alt="Processed Output" width="300" />}
    </div>
  );
}

export default Home;
