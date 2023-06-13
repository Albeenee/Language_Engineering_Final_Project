import React from 'react';
import {Link} from 'react-router-dom';

import classes from './Button.module.css';

const Button = props => {
    return(
        <div className={classes.choice}>
        <nav>
        <ul className={classes.list}>
        {props.items.map(b=> <li>
            <Link className={
                `${b.model==='trigram' ? classes.trigram:''} 
                ${b.model==='rnn' ? classes.rnn:''} 
                ${b.model==='lstm' ? classes.lstm:''}`} 
                to={b.link}>{b.name}</Link>
        </li>)}
        </ul>
        </nav>
    </div>
    )
};

export default Button