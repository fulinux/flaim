//------------------------------------------------------------------------------
// Desc:	Remove database test
//
// Tabs:	3
//
//		Copyright (c) 2006 Novell, Inc. All Rights Reserved.
//
//		This program is free software; you can redistribute it and/or
//		modify it under the terms of version 2 of the GNU General Public
//		License as published by the Free Software Foundation.
//
//		This program is distributed in the hope that it will be useful,
//		but WITHOUT ANY WARRANTY; without even the implied warranty of
//		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//		GNU General Public License for more details.
//
//		You should have received a copy of the GNU General Public License
//		along with this program; if not, contact Novell, Inc.
//
//		To contact Novell about this file by physical or electronic mail,
//		you may find current contact information at www.novell.com
//
// $Id$
//------------------------------------------------------------------------------

using System;
using System.IO;
using System.Runtime.InteropServices;
using xflaim;

namespace cstest
{

	//--------------------------------------------------------------------------
	// Remove database test.
	//--------------------------------------------------------------------------
	public class RemoveDbTest : Tester
	{
		public bool removeDbTest(
			string	sDbName,
			DbSystem	dbSystem)
		{
			beginTest( "Remove Database Test (" + sDbName + ")");
			try
			{
				dbSystem.dbRemove( sDbName, null, null, true);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "removing database");
				return( false);
			}
			endTest( false, true);
			return( true);
		}
	}
}
