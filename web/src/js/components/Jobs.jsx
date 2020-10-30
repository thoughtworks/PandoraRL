import React, {Component} from "react";
import PropTypes from "prop-types";
import JSONPretty from 'react-json-pretty';
import {Button} from "react-bootstrap";
import {ARTIFACTS_URL} from "../constants/urls";


export default class Jobs extends Component {
    constructor(props) {
        super(props);
        this.state = {
            selectedLog: undefined
        }
    }


    componentDidMount() {
        const {loadJobs} = this.props;
        loadJobs();
    }

render() {
    const {jobs} = this.props;
    return <div className="logs-tab">
        <label>Jobs</label>
        {jobs? Object.keys(jobs).map(key =>
            // <div key={key}>{JSON.stringify(jobs[key])}</div>
            <ul>
                <li>{`Job_${key}`}</li>
                <ul>
                    <li>Protein Input File: {jobs[key].protein_file_name} </li>
                    <li>Ligand Input File: {jobs[key].ligand_file_name} </li>
                    <li>Output File: 
                        <a href={ARTIFACTS_URL+jobs[key].output_path}>
                            {jobs[key].output_path.split("/").slice(-1)[0]}
                        </a>
                    </li>
                </ul>
            </ul>
        ):""}
    </div>
}
}


Jobs.propTypes = {
    loadJobs: PropTypes.func,
    jobs: PropTypes.object
};