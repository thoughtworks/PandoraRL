import axios from "axios";

export default function fetch(obj) {
  return axios(obj)
    .then(response => ({ response }))
    .catch(error => ({ error }));
}
