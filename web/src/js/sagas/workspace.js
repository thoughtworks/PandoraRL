import { all, call, put, takeEvery } from "redux-saga/effects";
import fetch from "../fetch";
import {handleRLAgentError} from "../actions/workspace";
export const HOST_URL = location.protocol + '//' + location.host;
export const BASE_URL = "/api/";
export const RL_AGENT_URL = HOST_URL+BASE_URL+"drugDiscoveryAgent/";

export function* triggerRLAgent({ inputFiles }) {
    const { response, error } = yield call(fetch, {
        url: RL_AGENT_URL,
        method: "post",
        data: inputFiles,
        headers: {
            "content-type": "multipart/form-data"
        }
    });
    if (response) {
        alert("success")
        // yield put(experimentArchivedSuccess());
    } else {
        alert("fail")
        const errorMessage = error.response ? error.response.data.message: "error occured";
        yield put(handleRLAgentError(errorMessage));
    }
}
export default function* workspaceSaga() {
    yield all([
        takeEvery("triggerRLAgent", triggerRLAgent)
    ]);
}
