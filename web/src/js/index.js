import React from "react";
import ReactDOM from "react-dom";
import {Provider} from "react-redux";
import {Router} from "react-router-dom";

import {store, sagaMiddleware} from "./reducers/index";
import rootSaga from "sagas/index";
import Routes from "routes/Routes";
import history from "util/history";
import "../assets/stylesheets/main.scss";
import "react-web-tabs/dist/react-web-tabs.css"
import 'react-table/react-table.css'

sagaMiddleware.run(rootSaga);
ReactDOM.render(
    <Provider store={store}>
        <Router history={history}>
            <Routes/>
        </Router>
    </Provider>,
    document.getElementById("root")

);
