import {all} from "redux-saga/effects";
import workspaceSaga from "./workspace";

export default function* rootSaga() {
    yield all([workspaceSaga()]);
}