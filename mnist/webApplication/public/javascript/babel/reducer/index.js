'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _redux = require('redux');

var _reducer = require('./reducer1.js');

/**
 * Created by leslie on 2017/7/19.
 */
var allReducers = (0, _redux.combineReducers)({
  users: _reducer.users

});

exports.default = allReducers;