/**
 * Created by leslie on 2017/7/19.
 */
import {combineReducers} from 'redux';
import {users} from './reducer1.js';

const allReducers = combineReducers({
    users: users

});

export default allReducers;