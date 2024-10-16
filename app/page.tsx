/* eslint-disable @typescript-eslint/no-unused-vars */
"use client"

import React, { useState } from 'react';
import { inferenceSqueezenet } from '@/utils/predict';
import { maskImage, postprocessImage } from '@/utils/imageHelper';
import applyMask from '@/actions/mask_image';

const Home: React.FC = () => {
  const [inputImage, setInputImage] = useState<ImageData | null>(null);
  const [outputImage, setOutputImage] = useState<ImageData | null>(null);
  const [outputSrc, setOutputSrc] = useState<string | null>(null);


  const saveInferenceResultToJson = (result: Float32Array, fileName: string) => {
    // JSON formatına dönüştür
    const jsonData = JSON.stringify(result, null, 2); // Pretty print

    // JSON verisini Blob olarak oluştur
    const blob = new Blob([jsonData], { type: 'application/json' });

    // Blob'dan URL oluştur
    const url = URL.createObjectURL(blob);

    // Dosya indirmek için bir link oluştur
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName; // İndirilecek dosya adı
    document.body.appendChild(a); // Linki DOM'a ekle
    a.click(); // Linke tıkla
    document.body.removeChild(a); // Linki DOM'dan kaldır
    URL.revokeObjectURL(url); // URL'yi serbest bırak

    console.log(`Inference result saved as ${fileName}`);
  };

  const runModel = async (imagePath: string) => {
    const image = new Image();
    image.src = imagePath;
    image.crossOrigin = 'Anonymous'; // CORS ayarları

    image.onload = async () => {
      const [inferenceResult, inferenceTime] = await inferenceSqueezenet(image);

      // Sonuçları konsola yazdır
      console.log(inferenceResult);
      console.log(inferenceTime);

      // Maskeyi işleme
      const mask = postprocessImage(inferenceResult, [image.height, image.width]);


      const originalImageData = await getImageData(image);

      const maskCanvas = document.createElement('canvas');
      const maskContext = maskCanvas.getContext('2d');

      if (!maskContext) return;

      // Maske resmini yükle
      maskCanvas.width = image.width;
      maskCanvas.height = image.height;
      maskContext.putImageData(mask, 0, 0);

      // Sunucu aksiyonunu tetikleme
      try {
        const finalImage = await maskImage(originalImageData, maskCanvas.toDataURL());

        const outputCanvas = document.createElement('canvas');
        const outputContext = outputCanvas.getContext('2d');

        if (!outputContext) return;

        // Maske resmini yükle
        outputCanvas.width = image.width;
        outputCanvas.height = image.height;
        const imageData = new ImageData(new Uint8ClampedArray(finalImage), image.width, image.height);
        outputContext.putImageData(imageData, 0, 0);
        setOutputSrc(outputCanvas.toDataURL());
        console.log(finalImage);
        // finalImage ile istediğiniz işlemleri gerçekleştirin
      } catch (error) {
        console.error("Mask image işlemi sırasında hata oluştu:", error);
      }
    };
  };

  // image'den Uint32Array elde etmek için yardımcı fonksiyon
  async function getImageData(image: HTMLImageElement): Promise<string> {
    // Bir canvas oluştur
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (ctx === null)
      throw new Error("Canvas context creation failed");

    // Canvas boyutunu resmi ayarla
    canvas.width = image.width;
    canvas.height = image.height;

    // Resmi canvas'a çiz
    ctx.drawImage(image, 0, 0);

    // ImageData elde et
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    // ImageData'yı Uint32Array'e dönüştür
    const uint32Array = new Float32Array(imageData.buffer);
    return canvas.toDataURL();
  }



  return (
    <div>
      <h1>ONNX Model Demo</h1>
      <button onClick={() => runModel("https://letsenhance.io/static/8f5e523ee6b2479e26ecc91b9c25261e/1015f/MainAfter.jpg")}>Launch</button>
      {outputSrc && <img src={outputSrc} alt="Processed Output" width="300" />}
      {inputImage && (
        <canvas
          width={inputImage.width}
          height={inputImage.height}
          ref={canvas => {
            if (canvas) {
              const ctx = canvas.getContext('2d');
              ctx?.putImageData(inputImage, 0, 0);
            }
          }}
        />
      )}
      {outputImage && (
        <canvas
          width={outputImage.width}
          height={outputImage.height}
          ref={canvas => {
            if (canvas) {
              const ctx = canvas.getContext('2d');
              ctx?.putImageData(outputImage, 0, 0);
            }
          }}
        />
      )}
    </div>
  );
}

export default Home;
