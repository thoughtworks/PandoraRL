import React from "react";
import { withRouter, Switch, Route } from "react-router-dom";
import Workspace from "../containers/workspace";

export class Routes extends React.Component {
  render() {
    return (
      <Switch>
        <Route exact path="/" component={Workspace} />
      </Switch>
    );
  }
}

export default withRouter(Routes);