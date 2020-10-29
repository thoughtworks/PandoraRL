const rootUrlElement = document.getElementById("rootUrl");
export const ROOT_URL = rootUrlElement ? rootUrlElement.value : '';

// export const HOST_URL = location.protocol + '//' + location.host;
// export const BASE_URL = `${ROOT_URL}/api/`;
// export const ARTIFACT_URL = `${ROOT_URL}/artifact`;
// export const HEALTHCHECK = HOST_URL + BASE_URL + "healthcheck/";
// export const WORKSPACE = HOST_URL + BASE_URL + "workspace/";
// export const EXPERIMENT = WORKSPACE + "experiment";
// export const PIPELINE = (uuid, pipelineUUID) => `${EXPERIMENT}/${uuid}/pipeline/${pipelineUUID}`;
