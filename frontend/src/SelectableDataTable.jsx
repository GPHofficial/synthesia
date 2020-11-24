import { forEach } from "lodash";
import { endianness } from "os";
import React, { useState, useEffect, ReactText, ReactElement } from "react"
import {
  ArrowTable,
  ComponentProps,
  Streamlit,
  withStreamlitConnection,
} from "streamlit-component-lib"

import './app.css'


// var selected: { [time: number] : {number: string}; } = {};

class Synthesia extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      dimensions: props.incoming.args.data.dimensions,
      selected: [[]],
      page: 0,
      generated: false
    }
    if(!('input' in props.incoming.args.data)){
      for(var i=0;i<this.state.dimensions[0];i++){
        this.state.selected[i] = ['white'];
        for(var j=0;j<this.state.dimensions[1];j++){
            this.state.selected[i][j] = 'white';
        }
      }
    }else{
      this.state.selected = this.decapsulateInput(props.incoming.args.data.input,this.state.page);
      this.state.generated = true
    }
    Streamlit.setComponentReady()
  }

  decapsulateInput(input,page){
    var grid = [];
    var start = page*this.state.dimensions[0]
    var end = (page+1)*this.state.dimensions[0]
    var standingNotes = [];
    var iteratorStart = (start - 50) <= 0 ? 0 : (start - 50);
    for(var i = iteratorStart;i<end;i++){
      //notes for current row
      var inRange = (i >= start);
      
      
      // concat new and existing rows
      if(i in input){
        standingNotes = standingNotes.concat(input[i]);
      }

      var rowNotes = []
      if(inRange){
        for(var j=0;j<this.state.dimensions[1];j++){
          rowNotes[j] = 'white';
        }
      }

      // add info to row
      var pendingRemovalIndex = []
      for(var j=0;j<standingNotes.length;j++){
        
        standingNotes[j][2] = standingNotes[j][2] - 1;
        if(standingNotes[j][2] <= 0){
          pendingRemovalIndex.push(j);
        }
        if(inRange){
          var currentNote = standingNotes[j][1]
          if(rowNotes[currentNote] != 'white'){
            if(rowNotes[currentNote] == 'blue' && standingNotes[j][0] == 'v'){
              rowNotes[currentNote] = 'magenta'
            }else if(rowNotes[currentNote] == 'red' && standingNotes[j][0] == 'p'){
              rowNotes[currentNote] = 'magenta'
            }
          }else{
            if(standingNotes[j][0] == 'v'){
              rowNotes[currentNote] = 'red'
            }else if(standingNotes[j][0] == 'p'){
              rowNotes[currentNote] = 'blue'
            }
          }
        }
      }

    //   //row cleanup
      for (var k = pendingRemovalIndex.length -1; k >= 0; k--)
        standingNotes = standingNotes.splice(pendingRemovalIndex[k],1);
    //   // currentArray
    //   // ["v", 43, 2]
      if(inRange)
        grid.push(rowNotes)
    }
    return grid;

  }

  // useEffect(() => {
  //   const interval = setInterval(() => {
  //     setSeconds(seconds => seconds + 1);
  //   }, 1000);
  //   return () => clearInterval(interval);
  // }, []);

  handleGridClick(value){
    
    var note = value[1];
    var time = value[0]
    var currentColor = this.state.selected[time][note];
    if(currentColor == 'white'){
      currentColor = 'blue'
    }else if(currentColor == 'blue'){
        currentColor = 'red'
    }else if(currentColor == 'red'){
      currentColor = 'magenta'
    }else if(currentColor == 'magenta'){
        currentColor = 'white'
    }

    var all = this.state.selected;
    all[time][note] = currentColor;
    this.setState({
      selected: all
    })
  }

  clear(){
    var empty = [[]]
    for(var i=0;i<this.state.dimensions[0];i++){
      empty[i] = ['white'];
      for(var j=0;j<this.state.dimensions[1];j++){
          empty[i][j] = 'white';
      }
    }
    this.setState({
      selected: empty
    })
  }

  next(){
    if(!this.state.generated)
      return;
    this.setState({
      selected: this.decapsulateInput(this.props.incoming.args.data.input,this.state.page + 1),
      page: this.state.page + 1
    })
  }

  previous(){
    if(!this.state.generated)
      return;
    if(this.state.page == 0)
      return;
    this.setState({
      selected: this.decapsulateInput(this.props.incoming.args.data.input,this.state.page - 1),
      page: this.state.page - 1
    })
  }
  

  getRow(count, row){
    row = this.state.dimensions[0]-1-row
    let item = 
        <tr>
            { 
              Array(count).fill(1).map((input,i)=>(<td onClick={() => this.handleGridClick([row,i])} style={{ backgroundColor: this.state.selected[row][i] }} ></td>))
            }
        </tr>
    return item;
  }

  generate(){
    var current = this.state.selected;
    var keys = []
    for(var i = 0; i < this.state.dimensions[0]; i++){
      for(var j = 0; j < this.state.dimensions[1]; j++){
        if(current[i][j] != 'white'){
          if(current[i][j] == 'blue')
            keys.push('p' + j)
          if(current[i][j] == 'red')
            keys.push('v' + j)
          if(current[i][j] == 'magenta'){
            keys.push('v' + j)
            keys.push('p' + j)
          }
        }
      }
      keys.push('wait1')
    }
    Streamlit.setComponentValue(keys);
  }

  render(){
    var buttonSet = <div></div>
    if(!this.state.generated){
      buttonSet = <div  style={{marginLeft: "10%", marginTop: "12px"}}><button onClick={() => this.generate()}>Generate</button><button style={{marginLeft: "20px"}} onClick={() => this.clear()}>Clear</button></div>
    }else{
      buttonSet = <div  style={{marginLeft: "10%", marginTop: "12px"}}><button onClick={() => this.previous()}>Previous</button>
        <button onClick={() => this.next()}  style={{marginLeft: "20px"}}>Next</button></div>
    }
    return (
      <div>
        <table>
          <tbody>
              { 
                Array.from(Array(this.state.dimensions[0]).keys()).reverse().map((input,i)=>(this.getRow(62,i)))
              }
              <tr>
              { 
                Array(this.state.dimensions[1]).fill(1).map((input,i)=>(<td>{i}</td>))
              }
              </tr>
              
              </tbody>
        </table>
        {
          buttonSet
        }
        {/* <button onClick={() => this.generate()}  display={this.state.generated?"none":""}>Generate</button>
        <button onClick={() => this.clear()} display={this.state.generated?"none":""} >Clear</button>
        <button onClick={() => this.previous()} display={!this.state.generated?"none":""} >Previous</button>
        <button onClick={() => this.next()} display={!this.state.generated?"none":""} >Next</button> */}
      </div>
    )
  }

  
  // static componentWillReceiveProps(nextProps, prevState){
    
  //   console.log("1")
  //   // console.log(nextProps)
  //   // console.log("11")
  //   // console.log(nextProps.incoming.args.data)
  //   // setState
  // }

  // playRow(data){
  //   if(!this.state.generated)
  //     return;

  //   var newRow = data.pop();
  //   var current = this.state.selected;
  //   data = []
  //   for(var j=0;j<this.state.dimensions[1];j++){
  //     data[j] = 'white';
  //   }
  //   current.pop();
  //   current.unshift(data);
  //   this.setState({
  //     selected: current
  //   })
  // }

}

function SelectableDataTable(props){

  useEffect(() => {
    var rowSize = (props.args.data.dimensions[0] + 1) * 25
    Streamlit.setFrameHeight(rowSize + 50)
  })

  return <div><Synthesia incoming={props}/></div>
}

export default withStreamlitConnection(SelectableDataTable)
