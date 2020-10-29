import createReducer from "./helper";
import {ToastStore} from 'react-toasts';

const initialState = {
    errorMessage: undefined,
};

const handleRLAgentError = (state, {errorMessage}) => {
    ToastStore.error(errorMessage);
    return {
        ...state,
        errorMessage: errorMessage
    }
};
const workspace = createReducer(initialState, {
    ["handleRLAgentError"]: handleRLAgentError,
});
export default workspace;
