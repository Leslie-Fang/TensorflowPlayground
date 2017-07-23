'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.store = undefined;

var _redux = require('redux');

var _index = require('../babel/reducer/index.js');

var _index2 = _interopRequireDefault(_index);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Created by leslie on 2017/7/19.
 */
var store = exports.store = (0, _redux.createStore)(_index2.default);