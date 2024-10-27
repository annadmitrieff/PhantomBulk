# Record parameters
        param_dict = asdict(params)
        param_dict['simulation_id'] = i
        
        # Convert planets list to string representation for DataFrame storage
        if params.planets:
            param_dict['planets'] = str([asdict(planet) for planet in params.planets])
        else:
            param_dict['planets'] = ''
            
        param_records.append(param_dict)
    
    # Save parameter database
    df = pd.DataFrame(param_records)
    df.to_csv(base_dir / 'parameter_database.csv', index=False)
    
    # Create interactive visualizations
    create_parameter_visualizations(df, base_dir)
    
    print(f"Generated {n_discs} disc configurations")
    print(f"Total number of planets: {sum(len(eval(row['planets'])) for row in df['planets'] if row)}")
    print(f"Files saved in: {base_dir}")
    print("Interactive visualizations have been created as HTML files in the output directory")

if __name__ == "__main__":
    main()

