import { all, call, put, takeEvery } from "redux-saga/effects";
import fetch from "../fetch";
import {handleRLAgentError} from "../actions/workspace";
import {JOBS_URL, LOGS_URL, RL_AGENT_URL} from "../constants/urls";
import {onLoadLogsSuccess, onLoadJobsSuccess, OnLoadJobs} from "actions/workspace";
import {ToastStore} from "react-toasts";

export function* triggerRLAgent({ inputFiles}) {
    const { response, error } = yield call(fetch, {
        url: RL_AGENT_URL,
        method: "post",
        data: inputFiles,
        headers: {
            "content-type": "multipart/form-data"
        }
    });
    if (response) {
        const jobId = response.data.job_id;
        yield put(OnLoadJobs());
        ToastStore.success("The job has been successfully submitted with job id "+ jobId);
    } else {
        const errorMessage = error.response ? error.response.data.message: "error occured";
        yield put(handleRLAgentError(errorMessage));
    }
}

export function* loadLogs() {
    const { response, error } = yield call(fetch, {
        url: LOGS_URL,
        method: "get"
    });
    if (response) {
        const logs = response.data;
        yield put(onLoadLogsSuccess(logs));
    }
}


export function* handleloadJobs() {
    const { response, error } = yield call(fetch, {
        url: JOBS_URL,
        method: "get"
    });
    if (response) {
        const jobs = response.data;
        yield put(onLoadJobsSuccess(jobs));
    } else {
        // alert("fail")
        // const errorMessage = error.response ? error.response.data.message: "error occured";
        // yield put(handleRLAgentError(errorMessage));
    }
}


export default function* workspaceSaga() {
    yield all([
        takeEvery("triggerRLAgent", triggerRLAgent),
        takeEvery("loadLogs", loadLogs),
        takeEvery("OnLoadJobs", handleloadJobs)
    ]);
}
