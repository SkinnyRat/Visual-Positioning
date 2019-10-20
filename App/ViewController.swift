//
//  ViewController.swift
//  CV
//
//  Created by ThisUser on 8/6/19.
//  Copyright © 2019 ThisUser. All rights reserved.
//

import UIKit

class ViewController: UIViewController, UINavigationControllerDelegate, UIImagePickerControllerDelegate {

    @IBOutlet var imageView: UIImageView!
    @IBOutlet weak var MainText: UILabel!
    var camera: Camera!
    var btnPressed: Int32!
    var theTitle: Float32!
    
    // Initialize Camera when the view loads
    override func viewDidLoad() {
        super.viewDidLoad()
        theTitle   = 10.0
        btnPressed = 1
        camera     = Camera(controller: self as? UIViewController & CameraDelegate, andImageView: imageView, titleText: &theTitle, bPressed: &btnPressed)
    }
    
    // Start it when it appears
    override func viewDidAppear(_ animated: Bool) {
        camera.start()
    }
    
    // Stop it when it disappears
    override func viewWillDisappear(_ animated: Bool) {
        camera.stop()
    }
    
    // Dispose of any resources that can be recreated.
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    // Update main label text at top.
    func updateMaintext() {
        let S = NSString(format: "Distance: %.1fm", theTitle)
        self.MainText.text = S as String
    }
    
    var imagePicker: UIImagePickerController!
    
    @IBAction func takePhoto(_ sender: Any) {
        updateMaintext()
        btnPressed += 1
    }
    
}




