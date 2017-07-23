/**
 * Created by leslie on 2017/7/19.
 */
import {createStore, applyMiddleware} from 'redux';
import allReducers from '../babel/reducer/index.js';

export const store = createStore(allReducers);