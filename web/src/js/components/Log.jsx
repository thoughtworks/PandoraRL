import React, {Component} from "react";
import PropTypes from "prop-types";
import JSONPretty from 'react-json-pretty';
import {Button} from "react-bootstrap";


export default class Logs extends Component {
    constructor(props) {
        super(props);
        this.state = {
            selectedLog: undefined
        }
    }

    showLogs(global = false) {
        const {loadLogs} = this.props;
        const {selectedLog} = this.state;

        if (!selectedLog || global) {
            this.setState({selectedLog: undefined});
            loadLogs();
            return;

        }

        loadLogs(this.state.selectedLog);
    }

    componentDidMount() {
        this.showLogs();
    }

    render() {
        const {logs, loadLogsError} = this.props;
        return <div className="logs-tab">
            <label>Logs</label>
            {loadLogsError ? <span className={"error"}> {loadLogsError} </span> : ""}
            <div className="logs-container">
                {logs ?
                    <JSONPretty data={logs.data}></JSONPretty> : loadLogsError ? "" : "No logs available"}
            </div>
            <div className="cube-refresh-logs-btn-section">
                <Button className="btn-primary btn-refresh-logs" onClick={() => this.showLogs()}>Load Logs</Button>
            </div>
        </div>
    }
}

Logs.propTypes = {
    loadLogs: PropTypes.func,
    logs: PropTypes.object,
    loadLogsError: PropTypes.string
};