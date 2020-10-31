import React, {Component, Fragment} from "react";
import {Button, Form, Grid, Modal, Col} from "react-bootstrap";
import PropTypes from "prop-types";
import Logs from "components/Log";
import Jobs from "components/Jobs";
import {ToastContainer, ToastStore} from "react-toasts";


export default class Workspace extends Component {

    constructor(props) {
        super(props);
        this.state = {
            proteinFile: undefined,
            ligandFile: undefined,
            smiles_string:false,
            ligandInput:""
        };
    }

    onSelectChange = (event, field) => {
        this.setState({[field]: event.target.files[0]});
    };

    onFormSubmit = () =>{
        const {onFileSubmit}= this.props;
        let input = new FormData();
        input.append("proteinFile", this.state.proteinFile);
        if(this.state.smiles_string!==true)
            input.append("ligandFile", this.state.ligandFile);
        else{
            input.append("ligandInput", this.state.ligandInput);
        }
        input.append("smiles_string",this.state.smiles_string)
        onFileSubmit(input)
    }

    isSubmitDisabled() {
        if ((this.state.proteinFile) && ((this.state.smiles_string && this.state.ligandInput) || (this.state.ligandFile))) {
            return false;
        }
        return true;
    }

    handleInputChange = (event) =>{
        const target = event.target;
        const value = target.type === 'checkbox' ? target.checked : target.value;
        const name = target.name;

        this.setState({
            [name]: value
        });
    }

    render() {
        const {loadLogsError, loadLogs, logs, jobs, OnLoadJobs}= this.props
        return (
            <Fragment>
                <div className="dashboard-header">
                    <h3 className="header">RL Based Ligand Pose Prediction</h3>
                </div>
                <Form name="ddh-Form" className="panel">
                    <div className="panel-title">Pose Prediction</div>
                    <div className="add-user-attachment">
                        <div className={"flex-display"}><h4>Ligand File</h4>
                        <label className={"text-style"}>[Accepted Format: .pdbqt or mol2]</label></div>
                        <label> Smiles String:
                            <input name="smiles_string" type="checkbox" checked={this.state.smiles_string}
                                   onChange={this.handleInputChange} />
                            {this.state.smiles_string?
                                <input name="ligandInput" className="input-ligand"
                                       id="ligandinput" type="text" onChange={this.handleInputChange}/>:""}
                        </label>
                        {!this.state.smiles_string?<input className="new-user-attachment" id="ligandinputpath" type="file"
                               onChange={e => this.onSelectChange(e, "ligandFile")}/>:""}
                    </div>

                    <div className="add-user-attachment">
                        <div className={"flex-display"}><h4>Prepared protein</h4>
                            <label className={"text-style"}>[Accepted Format: .pdbqt or mol2]</label></div>
                        <input className="new-user-attachment" id="preparedproteinpath" type="file"
                               onChange={e => this.onSelectChange(e, "proteinFile")}/>

                    </div>
                    <Button className='btn-primary'
                            onClick={this.onFormSubmit}
                            disabled={this.isSubmitDisabled()}>Generate Ligand Pose</Button>
                </Form>
                <Jobs loadJobs={OnLoadJobs} jobs={jobs}/>
                {/*todo - Log component removed for later*/}
                {/*<Logs loadLogs={loadLogs} logs={logs} loadLogsError={loadLogsError}/>*/}
                <ToastContainer store={ToastStore} position={ToastContainer.POSITION.TOP_CENTER}/>
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
