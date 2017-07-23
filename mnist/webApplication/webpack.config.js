var webpack = require('webpack');
const path = require('path');
module.exports = {
    entry: {
        main:'./public/javascript/babel/main.js',
    },
    output: {
        path : path.resolve(__dirname, 'public/javascript/webpack/'),
        filename: '[name].js'
    }
}