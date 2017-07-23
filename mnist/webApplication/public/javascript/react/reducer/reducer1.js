/**
 * Created by leslie on 2017/7/19.
 */
var userData=[
    {
        id:1,
        first:"Leslie",
        age:10
    },
    {
        id:3,
        first:"Bob",
        age:100
    }
];

export var users = function(state = userData,action){
    switch (action.type) {
        case 'SUBMIT_DATA':
            console.log('SUBMIT_DATA');
            console.log(action.payload);
            return Object.assign({},action.payload);
        case 'GET_VERYFY_DATA':
            console.log('GET_VERYFY_DATA');
            console.log(action.payload.state);
            if(action.payload.state == 'ok' ){
                window.location.href = '/voter';
            }
            else if(action.payload.state == 'novoter' ){
                window.location.href = '/';
                alert('无效的验证码');
            }
            else if(action.payload.state == 'secondvoter'){
                window.location.href = '/';
                alert('已经投过票了');
            }
            //window.location.href = '/voter';
            return Object.assign({},action.payload);
        default:
            console.log('return the defalut userData');
            return userData;
    }
};