const rootUrlElement = document.getElementById("rootUrl");
export const ROOT_URL = rootUrlElement ? rootUrlElement.value : '';

export const HOST_URL = location.protocol + '//' + location.host;
export const BASE_URL = "/api/";
export const RL_AGENT_URL = HOST_URL+BASE_URL+"drugDiscoveryAgent/";
export const LOGS_URL = RL_AGENT_URL+"logs";
export const JOBS_URL = RL_AGENT_URL+"jobs";
