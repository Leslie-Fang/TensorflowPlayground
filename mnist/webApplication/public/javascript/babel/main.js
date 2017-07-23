"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.Component1 = undefined;

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _reactRedux = require("react-redux");

var _store = require("../babel/store.js");

var _container = require("../babel/container/container1.js");

var _container2 = _interopRequireDefault(_container);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; } /**
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                * Created by leslie on 2017/7/19.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                */


var Component1 = exports.Component1 = function (_React$Component) {
    _inherits(Component1, _React$Component);

    function Component1(props) {
        _classCallCheck(this, Component1);

        return _possibleConstructorReturn(this, (Component1.__proto__ || Object.getPrototypeOf(Component1)).call(this, props));
    }

    _createClass(Component1, [{
        key: "render",
        value: function render() {
            return React.createElement(
                "div",
                null,
                React.createElement(
                    "h1",
                    { className: "text-center" },
                    "\u6295\u7968\u901A\u9053"
                )
            );
        }
    }]);

    return Component1;
}(React.Component);

;

var Board = function (_React$Component2) {
    _inherits(Board, _React$Component2);

    function Board(props) {
        _classCallCheck(this, Board);

        return _possibleConstructorReturn(this, (Board.__proto__ || Object.getPrototypeOf(Board)).call(this, props));
    }

    _createClass(Board, [{
        key: "render",
        value: function render() {
            return React.createElement(
                _reactRedux.Provider,
                { store: _store.store },
                React.createElement(
                    "div",
                    { className: "container" },
                    React.createElement(Component1, null),
                    React.createElement(_container2.default, null)
                )
            );
        }
    }]);

    return Board;
}(React.Component);

ReactDOM.render(React.createElement(Board, null), document.getElementById('main'));