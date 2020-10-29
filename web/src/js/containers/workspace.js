import { connect } from "react-redux";
import Workspace from "../components/Workspace";
import {triggerRLAgent} from "actions/workspace";

const mapStateToProps = state => ({
    workspace: state.workspace,
});
const mapDispatchToProps = dispatch => {
    return {
        onFileSubmit: (inputFiles)=> dispatch(triggerRLAgent(inputFiles))
    };
};
export default connect(
    mapStateToProps,
    mapDispatchToProps
)(Workspace);
