import { connect } from "react-redux";
import Workspace from "../components/Workspace";
import {OnLoadJobs, loadLogs, triggerRLAgent} from "actions/workspace";

const mapStateToProps = state => ({
    logs: state.workspace.logs,
    jobs: state.workspace.jobs,
    loadLogsError: state.workspace.loadLogsError

});
const mapDispatchToProps = dispatch => {
    return {
        onFileSubmit: (inputFiles)=> dispatch(triggerRLAgent(inputFiles)),
        loadLogs: ()=> dispatch(loadLogs()),
        OnLoadJobs: ()=> dispatch(OnLoadJobs())
    };
};
export default connect(
    mapStateToProps,
    mapDispatchToProps
)(Workspace);
