import createReducer from "./helper";
import {ToastStore} from 'react-toasts';

const initialState = {
    errorMessage: undefined,
    logs: undefined,
    jobs:undefined,
    loadLogsError:undefined
};

const handleRLAgentError = (state, {errorMessage}) => {
    ToastStore.error(errorMessage);
    return {
        ...state,
        errorMessage: errorMessage
    }
};

const onLoadLogsSuccess = (state, {logs}) => {
    return {
        ...state,
        logs: logs,
        loadLogsError: undefined
    }
};

const onLoadJobsSuccess = (state, {jobs}) => {
    return {
        ...state,
        jobs: jobs
    }
};
const workspace = createReducer(initialState, {
    ["handleRLAgentError"]: handleRLAgentError,
    ["onLoadLogsSuccess"]: onLoadLogsSuccess,
    ["onLoadJobsSuccess"]: onLoadJobsSuccess
});
export default workspace;
