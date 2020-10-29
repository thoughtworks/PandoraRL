const webpack = require("webpack"),
    HtmlWebpackPlugin = require("html-webpack-plugin"),
    ExtractTextPlugin = require("extract-text-webpack-plugin"),
    path = require("path");

const SRC = path.resolve(__dirname, "src"),
    NODE_MODULES = path.resolve(__dirname, "node_modules"),
    JS = path.resolve(__dirname, "src/js");

if (!process.env.NODE_ENV) {
    process.env.NODE_ENV = "development";
}
const config = {
    context: path.resolve(__dirname),
    devtool: "source-map",
    entry: './src/js/index.js',
    output: {
        path: __dirname + './../src/main/public/out',
        filename: 'bundle.js',
        publicPath: "/",
    },
    resolve: {
        extensions: [".js", ".jsx", ".css"],
        modules: [SRC, NODE_MODULES, JS]
    },
    module: {
        rules: [
            {
                test: /\.(js|jsx)$/,
                exclude: /node_modules/,
                use: 'babel-loader'
            },
            {
                test: /\.(css|sass|scss)$/,
                use: ExtractTextPlugin.extract({
                    fallback: "style-loader",
                    use: ["css-loader", "sass-loader"]
                })
            },
            {
                test: /\.(png|svg|jpg|gif|pdf)$/,
                use: [
                    {
                        loader: 'file-loader',
                        options: {
                            name: 'out/[name].[ext]',
                            outputPath: './../'
                        }
                    }
                ]
            }
        ]
    },
    plugins: [
        new ExtractTextPlugin({
            filename: "[name].css",
            disable: false,
            allChunks: true
        }),
        new HtmlWebpackPlugin({
            minify: {
                collapseWhitespace: true,
                removeComments: true
            },
            inject: false,
            template: "../src/main/public/index.html"
        }),
        new webpack.DefinePlugin({
            "process.env": {
                NODE_ENV: JSON.stringify(process.env.NODE_ENV || "development")
            }
        })
    ]
};

module.exports = config;
