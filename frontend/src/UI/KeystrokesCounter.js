import React from 'react';

import classes from './KeystrokesCounter.module.css';

const KeystrokesCounter = props => {
    return(
        <div className={classes.choice}>
        <ul className={classes.list}>
            <li className={props.model==='trigram' && classes.trigram || classes.rnn}>
                Saved Keystrokes : {props.results}
            </li>
        </ul>
    </div>
    )
};

export default KeystrokesCounter;