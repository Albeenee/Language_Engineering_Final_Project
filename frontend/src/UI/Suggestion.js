import React from 'react';

import classes from './Suggestion.module.css';

const Suggestion = props => {
    return(
        <div className={classes.results}>{props.results.map((result,index) => (
            <button 
            className={
                `${props.model==='trigram' ? classes.trigram:''} 
                ${props.model==='rnn' ? classes.rnn:''} 
                ${props.model==='lstm' ? classes.lstm:''}`} 
            key={index} 
            onClick={() => props.handlefunction(result)}>{result}</button>))}
        </div> 
    )
};

export default Suggestion;
