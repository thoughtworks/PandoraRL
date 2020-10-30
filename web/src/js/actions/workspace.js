import actionCreator from "./helper";

export const triggerRLAgent = actionCreator("triggerRLAgent", "inputFiles");
export const handleRLAgentError = actionCreator("handleRLAgentError", "errorMessage");
export const loadLogs = actionCreator("loadLogs");
export const OnLoadJobs = actionCreator("OnLoadJobs");
export const onLoadLogsSuccess = actionCreator("onLoadLogsSuccess", "logs");
export const onLoadJobsSuccess = actionCreator("onLoadJobsSuccess", "jobs");
