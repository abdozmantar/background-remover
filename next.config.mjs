/** @type {import('next').NextConfig} */
import NodePolyfillPlugin from "node-polyfill-webpack-plugin";
import CopyPlugin from "copy-webpack-plugin";

export const reactStrictMode = true;
export function webpack(config, { }) {

    config.resolve.extensions.push(".ts", ".tsx");
    config.resolve.fallback = { fs: false };

    config.plugins.push(
        new NodePolyfillPlugin(),
        new CopyPlugin({
            patterns: [
                {
                    from: './node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm',
                    to: 'static/chunks/pages',
                }, {
                    from: './node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs',
                    to: 'static/chunks/pages',
                },
                {
                    from: './model',
                    to: 'static/chunks/pages',
                },
            ],
        })
    );

    return config;
}
