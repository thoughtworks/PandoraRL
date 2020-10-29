import {combineReducers, createStore, applyMiddleware, compose} from "redux";
import createSagaMiddleware from "redux-saga";
import {composeWithDevTools} from "redux-devtools-extension";
import workspace from "./workspace";

export const sagaMiddleware = createSagaMiddleware();
const tool = process.env.NODE_ENV === "production" ? compose : composeWithDevTools;
export const store = createStore(combineReducers({
  workspace,
}), tool(applyMiddleware(sagaMiddleware)));
