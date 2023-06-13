import React from 'react';
import classes from './Form.module.css';

const Form = props => {
    return(
        <div className={classes.control}>
            <label htmlFor={props.html}></label>
            <input className={
                `${props.model==='trigram' ? classes.trigram:''} 
                ${props.model==='rnn' ? classes.rnn:''} 
                ${props.model==='lstm' ? classes.lstm:''}`}
              value={props.value}
              onChange={props.inputChangeHandlerFunction}
            />
        </div>
    )
};

export default Form;
