import {connect} from 'react-redux';
import {bindActionCreators} from 'redux';

function mapStateToProps(state) {
    return ({
        users: state.users
    });
}

function matchDispatchToProps(dispatch){
    return bindActionCreators({}, dispatch);
}

class Container1 extends React.Component {
    constructor(props) {
        super(props);
    }

    render() {
        return(
            <div>
                <form className = "text-center">
                    <div className="form-group">
                        <label for="exampleInputPassword1">验证码</label><br/>
                        <input type="text" className="form-control" id="exampleInputPassword1"
                               ref="verify"
                               value="hhhh" />
                    </div>
                    <button type="submit" className="btn btn-primary">登入</button>
                </form>
            </div>
        );
    }
}

//store.dispatch({ type: 'INCREMENT' });

export default connect(mapStateToProps,matchDispatchToProps)(Container1);