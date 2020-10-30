import React, {Component, Fragment} from "react";
import {Button, Form, Grid, Modal, Col} from "react-bootstrap";
import PropTypes from "prop-types";
import Logs from "components/Log";
import Jobs from "components/Jobs";


export default class Workspace extends Component {

    constructor(props) {
        super(props);
        this.state = {
            proteinFile: undefined,
            ligandFile: undefined,
        };
    }

    onSelectChange = (event, field) => {
        this.setState({[field]: event.target.files[0]});
    };

    onFormSubmit = () =>{
        const {onFileSubmit}= this.props;
        let inputFiles = new FormData();
        inputFiles.append("proteinFile", this.state.proteinFile);
        inputFiles.append("ligandFile", this.state.ligandFile);
        onFileSubmit(inputFiles)
    }

    isSubmitDisabled() {
        if (this.state.proteinFile && this.state.ligandFile) {
            return false;
        }
        return true;
    }

    render() {
        const {loadLogsError, loadLogs, logs, jobs, OnLoadJobs}= this.props
        return (
            <Fragment>
                <div className="dashboard-header">
                    <h3 className="header">RL based ligand pose prediction</h3>
                </div>
                <Form name="ddh-Form" className="panel">
                    <div className="panel-title">Pose Prediction</div>
                    <div className="add-user-attachment">
                        <h4>Ligand File</h4>
                        <input className="new-user-attachment" id="ligandinputpath" type="file"
                               onChange={e => this.onSelectChange(e, "ligandFile")}/>
                    </div>
                    <div className="add-user-attachment">
                        <h4>Prepared protein</h4>
                        <input className="new-user-attachment" id="preparedproteinpath" type="file"
                               onChange={e => this.onSelectChange(e, "proteinFile")}/>

                    </div>
                    <Button className='btn-primary'
                            onClick={this.onFormSubmit}
                            disabled={this.isSubmitDisabled()}>Submit</Button>
                </Form>
                <Jobs loadJobs={OnLoadJobs} jobs={jobs}/>
                <Logs loadLogs={loadLogs} logs={logs} loadLogsError={loadLogsError}/>
            </Fragment>
        );
    }
}

Workspace.propTypes = {
    onFileSubmit: PropTypes.func,
    loadLogs: PropTypes.func,
    OnLoadJobs: PropTypes.func,
    logs: PropTypes.object,
    jobs: PropTypes.object,
    loadLogsError: PropTypes.string
};
