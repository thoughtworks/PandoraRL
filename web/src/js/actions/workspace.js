import actionCreator from "./helper";

export const triggerRLAgent = actionCreator("triggerRLAgent", "inputFiles");
export const handleRLAgentError = actionCreator("handleRLAgentError", "errorMessage");